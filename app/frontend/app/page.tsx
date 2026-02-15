"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Terminal, Activity, Radio, Cpu, Download, 
  Search, Shield, Zap, Database, Play, Pause,
  Globe, Video, Layers, AlertCircle, X, Plus, Trash2,
  ListVideo, Youtube, Twitch, Film
} from "lucide-react";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts";
import axios from "axios";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// --- Utils ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Build embed URL for Twitch/YouTube to start at given timestamp (seconds). */
function embedUrlForSource(sourceUrl: string, startSeconds: number): string | null {
  try {
    if (sourceUrl.includes("twitch.tv")) {
      const m = sourceUrl.match(/twitch\.tv\/videos\/(\d+)/);
      if (m) {
        const videoId = m[1];
        const parent = typeof window !== "undefined" ? window.location.hostname : "localhost";
        return `https://player.twitch.tv/?video=${videoId}&t=${startSeconds}s&parent=${parent}`;
      }
    }
    if (sourceUrl.includes("youtube.com") || sourceUrl.includes("youtu.be")) {
      const m = sourceUrl.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/);
      if (m) {
        const videoId = m[1];
        return `https://www.youtube.com/embed/${videoId}?start=${Math.floor(startSeconds)}`;
      }
    }
  } catch {
    return null;
  }
  return null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getThumbnailUrl(sourceUrl: string): string | null {
  try {
     if (sourceUrl.includes("youtube.com") || sourceUrl.includes("youtu.be")) {
         const m = sourceUrl.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/);
         if (m) {
             return `https://img.youtube.com/vi/${m[1]}/hqdefault.jpg`;
         }
     }
     return null;
  } catch { return null; }
}

// --- Mock Data Generator ---
const generateChartData = () => {
  return Array.from({ length: 20 }, (_, i) => ({
    time: i,
    fps: Math.floor(Math.random() * 300) + 100,
    bandwidth: Math.floor(Math.random() * 150) + 50,
  }));
};

export default function Dashboard() {
  // --- State ---
  const [prompt, setPrompt] = useState("");
  const [url, setUrl] = useState("");
  const [source, setSource] = useState("Youtube");
  const [scale, setScale] = useState(10);
  const [stealth, setStealth] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionProgress, setIngestionProgress] = useState({ current: 0, total: 0, percent: 0, status: 'idle' });
  
  const [stats, setStats] = useState({ active_workers: 0, fps: 0, bandwidth: 0 });
  const [chartData, setChartData] = useState(generateChartData());
  const [logs, setLogs] = useState<string[]>([]);
  const [frames, setFrames] = useState<any[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Terminal State
  const [termHeight, setTermHeight] = useState(192); // Default h-48
  const [termSearch, setTermSearch] = useState("");
  const [isTermExpanded, setIsTermExpanded] = useState(false);
  const termRef = useRef<HTMLDivElement>(null);

  // PiP overlay: video over cluster starting at frame timestamp
  const [pipFrame, setPipFrame] = useState<{
    source_url: string;
    timestamp_seconds?: number;
    title: string;
  } | null>(null);

  // Tab State
  const [activeTab, setActiveTab] = useState<"analyze" | "create" | "import">("analyze");

  // Import Tab State
  const [importPrompt, setImportPrompt] = useState("");
  const [importedUrls, setImportedUrls] = useState<string[]>([]);
  const [isImporting, setIsImporting] = useState(false);

  // File Upload Handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (!text) return;
      
      const lines = text.split('\n');
      if (lines.length < 2) return;
      
      // Simple CSV parser that handles quotes
      const parseCSVLine = (line: string) => {
          const res = [];
          let current = '';
          let inQuotes = false;
          for (let i = 0; i < line.length; i++) {
              const char = line[i];
              if (char === '"') {
                  inQuotes = !inQuotes;
              } else if (char === ',' && !inQuotes) {
                  res.push(current);
                  current = '';
              } else {
                  current += char;
              }
          }
          res.push(current);
          return res.map(c => c.trim().replace(/^"|"$/g, ''));
      };

      const header = parseCSVLine(lines[0]);
      const urlIndex = header.findIndex(h => h.toLowerCase() === 'url');
      
      if (urlIndex === -1) {
          alert('CSV must contain a "url" column. Found: ' + header.join(', '));
          return;
      }

      const urls = lines.slice(1).map(line => {
          // Skip empty lines
          if (!line.trim()) return null;
          const columns = parseCSVLine(line);
          
          // Strategy 1: Check "url" column if it exists and has a valid URL
          if (urlIndex !== -1 && columns.length > urlIndex) {
             const val = columns[urlIndex]?.trim();
             // Clean up if it contains a pipe like "Title | URL"
             const pipeMatch = val.match(/\|\s*(https?:\/\/[^\s]+)/);
             if (pipeMatch) {
                 return pipeMatch[1];
             }
             
             if (val && (val.startsWith('http') || val.includes('youtube.com') || val.includes('youtu.be') || val.includes('twitch.tv'))) {
                 return val;
             }
          }

          // Strategy 2: Scan ALL columns for a valid video URL
          for (const col of columns) {
              const val = col.trim();
              
              // Check for "Title | URL" pattern specifically which seems to be common in your CSV
              const pipeMatch2 = val.match(/\|\s*(https?:\/\/[^\s]+)/);
              if (pipeMatch2) {
                 return pipeMatch2[1];
              }

              // Check if the cell itself IS a URL
              if (val.startsWith('http') || val.includes('youtube.com') || val.includes('youtu.be') || val.includes('twitch.tv')) {
                  return val;
              }
              // Check if the cell CONTAINS a URL 
              const urlMatch = val.match(/(https?:\/\/[^\s|]+)/); // Basic extraction
              if (urlMatch) {
                  const extracted = urlMatch[0];
                  // Verify it's actually a video link we care about
                  if (extracted.includes('youtube') || extracted.includes('youtu.be') || extracted.includes('twitch')) {
                      return extracted;
                  }
              }
          }
          return null;
      }).filter((u): u is string => !!u);
      
      console.log("Extracted URLs:", urls);
      if (urls.length === 0) {
        alert("No valid URLs found in the CSV. Make sure they are in a 'url' column and contain http/youtube/twitch links.");
      } else {
        setImportedUrls(urls);
        setLogs(prev => [...prev, `[INFO] Imported ${urls.length} URLs from CSV`]);
      }
    };
    reader.onerror = (error) => console.log('Error reading file:', error);
    reader.readAsText(file);
  };

  const handleImportDataset = async () => {
    // Maximize resources for batch processing
    setScale(100);
    setIsImporting(true);
    setDatasetFrames([]);
    setScrapeResults(null); 
    
    try {
      if (!importPrompt.trim()) {
        alert("Please describe the dataset.");
        setIsImporting(false);
        return;
      }
      
      if (importedUrls.length === 0) {
        alert("Please upload a CSV with URLs first.");
        setIsImporting(false);
        return;
      }

      setLogs(prev => [...prev, `[INFO] Starting Import Job: "${importPrompt}" [${importedUrls.length} Sources]...`]);

      const res = await axios.post(`${API_BASE}/dataset/create`, {
        prompt: importPrompt,
        urls: importedUrls,
        scale: 100,
      });

      setScrapeResults(res.data);
      setLogs(prev => [...prev, `[SUCCESS] Import Job Initiated: ${JSON.stringify(res.data.message)}`]);
      
      if (res.status === 200) {
         setIsMonitoringDataset(true);
      }

    } catch (e: any) {
      setLogs(prev => [...prev, `[ERROR] Import error: ${e.message}`]);
      setScrapeResults({ error: e.message });
      setIsMonitoringDataset(false);
    } finally {
      setIsImporting(false);
    }
  };

  // Create Tab State
  const [scrapeUrls, setScrapeUrls] = useState<string[]>([""]);
  const [scrapeResults, setScrapeResults] = useState<any>(null);
  const [isScraping, setIsScraping] = useState(false);
  const [createPrompt, setCreatePrompt] = useState("");
  const [datasetFrames, setDatasetFrames] = useState<any[]>([]);
  const [isMonitoringDataset, setIsMonitoringDataset] = useState(false);

  const fetchDatasetFrames = async () => {
    try {
      // Use the prompt corresponding to the active tab
      let currentPrompt = activeTab === "create" ? createPrompt : importPrompt;
      if (!currentPrompt) return; // Don't fetch if no prompt
      
      // Get URLs that are actually being processed (Twitch links in this demo)
      // or just all discovered URLs
      const targetUrls = scrapeResults?.jobs?.map((j: any) => j.url) || (activeTab === "import" ? importedUrls : []) || [];

      const res = await axios.post(`${API_BASE}/search`, {
          query: currentPrompt,
          top_k: 24,
          allowed_sources: targetUrls.length > 0 ? targetUrls : undefined
      });
      if (res.data.ok && res.data.results.length > 0) {
          setDatasetFrames(res.data.results);
      }
    } catch (e) {
        console.error("Dataset frame fetch failed", e);
    }
  };

  useEffect(() => {
    const currentPrompt = activeTab === "create" ? createPrompt : importPrompt;
    
    // Only fetch if monitoring is active AND we have a prompt
    if (!isMonitoringDataset || !currentPrompt) return;
    
    // Initial fetch
    fetchDatasetFrames();
    
    const interval = setInterval(() => fetchDatasetFrames(), 4000);
    return () => clearInterval(interval);
  }, [isMonitoringDataset, createPrompt, importPrompt, activeTab]);

  const handleCreateDataset = async () => {
    setIsScraping(true);
    setScrapeResults(null);
    setDatasetFrames([]); // Clear previous results
    
    try {
      const validUrls = scrapeUrls.filter((u) => u.trim() !== "");
      
      if (!createPrompt.trim()) {
        alert("Please describe the dataset.");
        setIsScraping(false);
        return;
      }

      addLog(`Starting Dataset Job: "${createPrompt}" [${validUrls.length > 0 ? validUrls.length + ' Sources' : 'Auto-Discovery'}]...`);

      const res = await axios.post(`${API_BASE}/dataset/create`, {
        prompt: createPrompt,
        urls: validUrls,
      });

      setScrapeResults(res.data);
      addLog(`Dataset Job Initiated: ${JSON.stringify(res.data.message)}`);
      
      // Start monitoring for results
      if (res.data.ok) {
        setIsMonitoringDataset(true);
      }

    } catch (e: any) {
      addLog(`Dataset error: ${e.message}`);
      setScrapeResults({ error: e.message });
      setIsMonitoringDataset(false);
    } finally {
      setIsScraping(false);
    }
  };

  // --- Effects ---
  
  // Real-time chart updates
  useEffect(() => {
    if (!isIngesting) return; // Don't run interval if not ingesting

    const interval = setInterval(async () => {
      // Fetch stats from backend
      try {
        const res = await axios.get(`${API_BASE}/stats`);
        if (res.data) {
           setStats({
             active_workers: res.data.active_workers || 0,
             fps: res.data.fps_processed || 0,
             bandwidth: res.data.bandwidth_mbps || 0
           });
        }

        // Poll Ingestion Progress if URL is set
        if (url) {
             const statusRes = await axios.get(`${API_BASE}/ingestion/status?url=${encodeURIComponent(url)}`);
             if (statusRes.data.status !== "not_found") {
                setIngestionProgress({
                  current: statusRes.data.processed_segments,
                  total: statusRes.data.total_segments,
                  percent: statusRes.data.progress * 100,
                  status: statusRes.data.status
                });
             }
        }
      } catch (e) {
        // Fallback to simulation if backend down
        setStats({
             active_workers: Math.min(scale, Math.floor(Math.random() * scale) + (scale/2)),
             fps: Math.floor(Math.random() * 100) + (scale * 20),
             bandwidth: Math.floor(Math.random() * 50) + (scale * 5),
        });
      }

      setChartData(prev => {
        const newData = [...prev.slice(1), {
          time: prev[prev.length - 1].time + 1,
          fps: Math.floor(Math.random() * 100) + (scale * 20),
          bandwidth: Math.floor(Math.random() * 50) + (scale * 5),
        }];
        return newData;
      });
      
      // Add Random Log if simulating (only occasionally)
      if (Math.random() > 0.85) {
        const actions = ["Ingesting", "Processing", "Vectorizing", "Uploading"];
        const vidId = Math.random().toString(36).substring(7);
        addLog(`[Worker-${Math.floor(Math.random() * scale)}] ${actions[Math.floor(Math.random() * 4)]} video_${vidId} ... 100%`);
      }

    }, 1000);
    return () => clearInterval(interval);
  }, [isIngesting, scale]);

  // Scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Handle Start
  const handleStart = async () => {
    setIsIngesting(true);
    setFrames([]); // Clear old frames immediately
    addLog(`INITIALIZING CLUSTER...`);
    addLog(`Source: ${source} | URL: ${url || "Default"} | Target: "${prompt}"`);
    addLog(`Scaling to ${scale} workers on A10G GPUs...`);
    if (stealth) addLog(`[STEALTH] Bright Data Resi-Proxies ENGAGED 🥷`);
    
    try {
        // Call the backend API
        const res = await axios.post(`${API_BASE}/ingest/start`, null, {
            params: {
              source_url: url || "https://www.twitch.tv/videos/2689445480", // Default from modal file
              prompt: prompt,
              scale: scale,
              stealth: stealth
            }
        });

        if (res.data.ok) {
           addLog(`✅ SUCCESS: ${res.data.message}`);
           addLog(`🆔 JOB ID: ${res.data.job_id}`);
           
           addLog(`⏳ WAITING FOR FIRST BATCH...`);
           
           // Wait until we actually have some data
           const checkDataInterval = setInterval(async () => {
              try {
                // Poll for at least one result
                const payload = { query: prompt, top_k: 1, source_url: url || undefined };
                const checkRes = await axios.post(`${API_BASE}/search`, payload);
                if (checkRes.data.ok && checkRes.data.results.length > 0) {
                    clearInterval(checkDataInterval);
                    addLog(`🚀 DATA STREAM ACTIVE - STARTING VISUALIZATION`);
                    fetchFrames(); // Initial fetch
                }
              } catch (e) {
                // Keep waiting
              }
           }, 2000);

        } else {
           addLog(`⚠️ CRITICAL FAILURE: ${res.data.message}`);
           setIsIngesting(false);
        }

    } catch (e: any) {
        addLog(`❌ FATAL ERROR: ${e.response?.data?.message || e.message}`);
        addLog(`🛑 ABORTING OPERATION`);
        setIsIngesting(false);
    }
  };

  const handleStop = () => {
    setIsIngesting(false);
    addLog(`CLUSTER SHUTDOWN INITIATED.`);
  };

  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const fetchFrames = async () => {
    try {
      const payload = {
         query: prompt, 
         top_k: 48, // Increased to get more candidates for grouping
         source_url: url || undefined
      };
      
      const res = await axios.post(`${API_BASE}/search`, payload);
      if (res.data.ok) {
        const results = res.data.results as Array<{ timestamp_seconds?: number; [k: string]: unknown }>;
        if (results.length === 0) {
          // Only clear if we really have no results and it's the first fetch? 
          // Or just keep old ones? Let's just update provided we aren't completely empty to avoid flashing
          if (frames.length === 0) setFrames([]);
          addLog(`Scanning for matches...`);
          return;
        }

        // Group timestamps into videos
        const times = results.map((r) => r.timestamp_seconds ?? 0).filter((t) => typeof t === "number");
        const videoLength = Math.max(...times, 0) + 120;
        
        try {
          const groupRes = await axios.post(`${API_BASE}/timestamps/group`, {
             times,
             video_length: videoLength,
          });
          
          const segments: number[][] = groupRes.data?.segments ?? [];
          
          if (segments.length === 0) {
             // Fallback to raw results if no segments found (rare)
             setFrames(results);
             return;
          }

          // Map Segments to Frames
          // refined: finding the frame that is closest to the *start* of the segment
          const consolidatedFrames = segments.map((seg) => {
             const [start, end] = seg;
             
             // All frames that loosely fall into this segment's range/vicinity
             // We expand range slightly to catch edge cases
             const candidates = results.filter(r => {
                const t = r.timestamp_seconds ?? 0;
                return t >= start - 1 && t <= end + 1;
             });

             // If candidates found, pick the one closest to 'start' timestamp
             let best = candidates.length > 0 
                ? candidates.reduce((a, b) => Math.abs((a.timestamp_seconds??0) - start) < Math.abs((b.timestamp_seconds??0) - start) ? a : b)
                : results.reduce((a, b) => Math.abs((a.timestamp_seconds??0) - start) < Math.abs((b.timestamp_seconds??0) - start) ? a : b); // fallback to global closest

              // Format Timestamp Range
              const fmt = (s: number) => {
                 const m = Math.floor(s / 60);
                 const sec = Math.floor(s % 60);
                 return `${m}:${sec.toString().padStart(2, '0')}`;
              };

             return {
                 ...best,
                 // Override visuals to represent the SEGMENT
                 timestamp_display: `${fmt(start)} - ${fmt(end)}`,
                 timestamp_seconds: start, // Play from start
                 title: `${best.title}`,
             };
          });

          // Dedupe by timestamp just in case
          const seen = new Set();
          const unique = consolidatedFrames.filter(f => {
              const key = `${f.title}-${f.timestamp_seconds}`;
              if (seen.has(key)) return false;
              seen.add(key);
              return true;
          });

          setFrames(unique);
          
        } catch (_) {
          // If grouping fails, show raw
          setFrames(results);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Poll for updates during ingestion
  useEffect(() => {
    if (isIngesting && prompt) {
        const i = setInterval(fetchFrames, 3000);
        return () => clearInterval(i);
    }
  }, [isIngesting, prompt, url]);

  return (
    <div className="flex flex-col h-screen w-full bg-[#09090b] text-zinc-300 font-mono overflow-hidden">
        {/* TOP BAR / TABS */}
        <div className="h-14 border-b border-zinc-900 bg-zinc-950 flex items-center px-6 justify-between flex-shrink-0">
             <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-neon-green rounded-full animate-pulse" />
                <span className="font-bold text-white tracking-widest text-lg">SHOTSPOT</span>
             </div>
             
             <div className="flex bg-zinc-900 p-1 rounded-lg">
                <button 
                    onClick={() => setActiveTab("analyze")}
                    className={cn(
                        "px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2",
                        activeTab === "analyze" ? "bg-zinc-800 text-white shadow-sm" : "text-zinc-500 hover:text-zinc-300"
                    )}
                >
                    <Activity size={14} /> ANALYZE DATA
                </button>
                <button 
                    onClick={() => setActiveTab("create")}
                    className={cn(
                        "px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2",
                        activeTab === "create" ? "bg-zinc-800 text-white shadow-sm" : "text-zinc-500 hover:text-zinc-300"
                    )}
                >
                    <Database size={14} /> CREATE DATA
                </button>
                <button 
                    onClick={() => setActiveTab("import")}
                    className={cn(
                        "px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2",
                        activeTab === "import" ? "bg-zinc-800 text-white shadow-sm" : "text-zinc-500 hover:text-zinc-300"
                    )}
                >
                    <Download size={14} /> IMPORT DATASET
                </button>
             </div>
             
             <div className="text-xs text-zinc-500">v0.1.0-alpha</div>
        </div>

        {/* MAIN CONTENT AREA */}
        <div className="flex-1 overflow-hidden relative">
          {activeTab === "create" ? (
             <div className="h-full w-full p-8 overflow-y-auto">
                 <div className="max-w-4xl mx-auto space-y-8">
                    <div className="space-y-2">
                        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                            <Database className="text-neon-green" /> Generative Dataset Construction
                        </h1>
                        <p className="text-zinc-500">
                            Describe the dataset you want to build using natural language. The system will autonomously discover high-relevance video sources via Bright Data, ingest the content, and construct a structured dataset matching your description.
                        </p>
                    </div>

                    <div className="glass-panel p-6 space-y-6">
                        <div className="space-y-2">
                            <h2 className="text-sm font-bold uppercase text-zinc-400">Dataset Description</h2>
                            <textarea 
                                value={createPrompt}
                                onChange={(e) => setCreatePrompt(e.target.value)}
                                placeholder="Describe the visual elements you want to extract (e.g., 'Rocket launches', 'People holding Red Bull cans', 'Specific dance moves')..."
                                className="w-full bg-zinc-900 border border-zinc-800 rounded p-3 text-sm text-white focus:outline-none focus:border-neon-green transition-colors min-h-[100px]"
                            />
                        </div>

                        <div className="flex justify-between items-center pt-4 border-t border-zinc-800">
                            <h2 className="text-sm font-bold uppercase text-zinc-400">
                                Source URLs <span className="text-zinc-600 normal-case text-xs">(Optional)</span>
                            </h2>
                            <button 
                                onClick={() => setScrapeUrls([...scrapeUrls, ""])}
                                className="text-xs bg-zinc-800 hover:bg-zinc-700 text-neon-green px-3 py-1.5 rounded font-bold flex items-center gap-1 transition-colors"
                            >
                                <Plus size={14} /> ADD SOURCE
                            </button>
                        </div>
                        
                        <div className="space-y-3">
                            {scrapeUrls.map((u, idx) => (
                                <div key={idx} className="flex gap-2">
                                    <input 
                                        type="text" 
                                        placeholder="https://www.twitch.tv/videos/..."
                                        className="flex-1 bg-zinc-900 border border-zinc-800 rounded p-3 text-sm text-white focus:outline-none focus:border-neon-green transition-colors"
                                        value={u}
                                        onChange={(e) => {
                                            const newUrls = [...scrapeUrls];
                                            newUrls[idx] = e.target.value;
                                            setScrapeUrls(newUrls);
                                        }}
                                    />
                                    {scrapeUrls.length > 1 && (
                                        <button 
                                            onClick={() => {
                                                const newUrls = [...scrapeUrls];
                                                newUrls.splice(idx, 1);
                                                setScrapeUrls(newUrls);
                                            }}
                                            className="p-3 bg-zinc-900 border border-zinc-800 rounded hover:border-red-500 hover:text-red-500 text-zinc-500 transition-colors"
                                        >
                                            <Trash2 size={18} />
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>

                        <div className="pt-4 border-t border-zinc-800 flex justify-end">
                            <button 
                                onClick={handleCreateDataset} 
                                disabled={isScraping}
                                className={cn(
                                    "px-8 py-3 rounded font-bold uppercase tracking-widest text-sm transition-all flex items-center gap-2",
                                    isScraping 
                                      ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                                      : "bg-neon-green text-black hover:bg-white hover:shadow-[0_0_20px_rgba(34,197,94,0.4)]"
                                )}
                            >
                                {isScraping ? <Activity className="animate-spin" /> : <Play size={18} fill="currentColor" />}
                                {isScraping ? "Initiating Job..." : "Create Dataset"}
                            </button>
                        </div>
                    </div>

                    {scrapeResults && (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                             <div className="flex justify-between items-center">
                                <h2 className="text-sm font-bold uppercase text-zinc-400">Job Status</h2>
                                <button 
                                    onClick={() => {
                                        window.open(`${API_BASE}/dataset/export?query=${encodeURIComponent(createPrompt)}`, '_blank');
                                    }}
                                    className="text-xs bg-neon-green/10 border border-neon-green hover:bg-neon-green/20 text-neon-green px-3 py-1.5 rounded font-bold flex items-center gap-2 transition-colors"
                                >
                                    <Download size={14} /> EXPORT DATASET (.ZIP)
                                </button>
                             </div>
                             
                             <div className="glass-panel p-0 overflow-hidden relative">
                                 <div className="absolute top-0 right-0 p-2">
                                     <button className="text-xs bg-zinc-800 hover:bg-zinc-700 text-white px-2 py-1 rounded"
                                        onClick={() => navigator.clipboard.writeText(JSON.stringify(scrapeResults, null, 2))}
                                     >
                                         Copy Info
                                     </button>
                                 </div>
                                 <pre className="p-4 text-xs font-mono text-zinc-300 overflow-x-auto max-h-[200px] border-b border-zinc-800">
                                     {JSON.stringify(scrapeResults, null, 2)}
                                 </pre>

                                 {/* NEW: Discovered Sources List */}
                                 {scrapeResults.jobs && scrapeResults.jobs.length > 0 && (
                                     <div className="p-4 bg-zinc-950/50">
                                         <h3 className="text-xs font-bold text-zinc-500 uppercase mb-3 flex items-center gap-2">
                                             <ListVideo size={14} /> Discovered Sources
                                         </h3>
                                         <div className="grid gap-2">
                                             {scrapeResults.jobs.map((job: any, i: number) => (
                                                 <div key={i} className="flex items-center gap-3 text-xs p-2 rounded bg-zinc-900/50 border border-zinc-800/50 hover:border-zinc-700 transition-colors group">
                                                     {/* Stack Icon */}
                                                     <div className={cn(
                                                         "w-2 h-2 rounded-full",
                                                         job.status === "started" ? "bg-neon-green animate-pulse" : 
                                                         job.status === "error" ? "bg-red-500" : "bg-yellow-500"
                                                     )} />
                                                     
                                                     {/* Platform Icon */}
                                                     {job.url.includes("youtube.com") || job.url.includes("youtu.be") ? (
                                                         <Youtube size={14} className="text-red-400" />
                                                     ) : (
                                                         <Twitch size={14} className="text-purple-400" />
                                                     )}

                                                     {/* Link */}
                                                     <a href={job.url} target="_blank" rel="noopener noreferrer" className="flex-1 truncate hover:text-white hover:underline decoration-zinc-600 underline-offset-4">
                                                         {job.url}
                                                     </a>

                                                     {/* Status Badge */}
                                                     <span className={cn(
                                                         "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
                                                         job.status === "started" ? "bg-neon-green/10 text-neon-green" : 
                                                         job.status === "error" ? "bg-red-500/10 text-red-500 border border-red-500/20" : "bg-yellow-500/10 text-yellow-500 border border-yellow-500/20"
                                                     )}>
                                                         {job.status === "started" ? "INGESTING" : job.status === "discovered_only" ? "DISCOVERED" : "ERROR"}
                                                     </span>
                                                     
                                                     {/* Job ID (if ingesting) */}
                                                     {job.job_id && (
                                                         <span className="text-[10px] text-zinc-600 font-mono hidden sm:inline-block">
                                                             ID: {job.job_id.substring(0,8)}...
                                                         </span>
                                                     )}
                                                 </div>
                                             ))}
                                         </div>
                                     </div>
                                 )}
                             </div>
                        </div>
                    )}

                    {/* Dataset Visual Explorer */}
                    {(datasetFrames.length > 0 || isMonitoringDataset) && (
                         <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-150">
                             <div className="flex justify-between items-center">
                                <h2 className="text-sm font-bold uppercase text-zinc-400 flex items-center">
                                    <Database className="inline mr-2 text-neon-green" size={16}/> Dataset Visual Explorer
                                </h2>
                                <div className="flex items-center gap-2">
                                   {isMonitoringDataset && (
                                     <span className="flex items-center gap-2 text-[10px] text-neon-green animate-pulse">
                                        <Activity size={12} /> LIVE UPDATING
                                     </span>
                                   )}
                                   <span className="text-xs bg-zinc-800 px-2 py-0.5 rounded text-zinc-400 font-mono">
                                       {datasetFrames.length} SAMPLES
                                   </span>
                                </div>
                             </div>
                             
                             {datasetFrames.length === 0 ? (
                                 <div className="glass-panel p-12 flex flex-col items-center justify-center text-zinc-500 space-y-4 border-dashed">
                                     <div className="relative">
                                         <div className="absolute inset-0 bg-neon-green/20 blur-xl rounded-full animate-pulse" />
                                         <Cpu size={48} className="relative z-10 animate-pulse text-neon-green" />
                                     </div>
                                     <div className="text-center space-y-1">
                                         <h3 className="text-white font-bold">Processing Video Streams...</h3>
                                         <p className="text-xs max-w-sm">
                                             Ingesting content, extracting frames, and embedding vectors. 
                                             New samples will appear here automatically.
                                         </p>
                                     </div>
                                 </div>
                             ) : (
                                 <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                                    {datasetFrames.map((frame, i) => (
                                        <div 
                                            key={i} 
                                            className="group relative aspect-video bg-black rounded-lg border border-zinc-800 overflow-hidden hover:border-neon-green transition-all cursor-pointer shadow-lg hover:shadow-neon-green/10"
                                            onClick={() => setPipFrame({
                                                source_url: frame.source_url,
                                                timestamp_seconds: frame.timestamp_seconds,
                                                title: frame.title
                                            })}
                                        >
                                            {/* Thumbnail / Loading State */}
                                            <div className="absolute inset-0 flex items-center justify-center bg-zinc-900 text-zinc-700">
                                                <Film size={24} />
                                            </div>
                                            
                                            {/* Thumbnail & Visuals */}
                                            {(() => {
                                                // 1. Try YouTube Thumbnail
                                                if (frame.source_url.includes('youtube.com') || frame.source_url.includes('youtu.be')) {
                                                    const vId = frame.source_url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/)?.[1];
                                                    if (vId) {
                                                        return (
                                                            <img 
                                                                src={`https://img.youtube.com/vi/${vId}/mqdefault.jpg`} 
                                                                className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity"
                                                                alt="Thumbnail"
                                                            />
                                                        );
                                                    }
                                                }
                                                
                                                // 2. Twitch / Other styling
                                                if (frame.source_url.includes('twitch.tv')) {
                                                    return (
                                                        <div className="absolute inset-0 w-full h-full bg-gradient-to-br from-purple-900/20 to-zinc-900 group-hover:from-purple-900/40 transition-all flex flex-col items-center justify-center gap-3">
                                                            {/* Stylized Glitch Effect Background */}
                                                            <div className="absolute inset-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] mix-blend-overlay"></div>
                                                            
                                                            <div className="relative z-10 p-4 rounded-full bg-black/40 backdrop-blur-sm border border-purple-500/30 shadow-[0_0_30px_rgba(168,85,247,0.2)] group-hover:scale-110 transition-transform duration-300">
                                                                <Twitch size={32} className="text-purple-400" />
                                                            </div>
                                                            <div className="flex flex-col items-center">
                                                                <span className="text-[10px] text-purple-300/70 font-bold uppercase tracking-[0.2em]">Twitch VOD</span>
                                                                <span className="text-[9px] text-purple-400/50 font-mono">Stream Recording</span>
                                                            </div>
                                                        </div>
                                                    )
                                                }

                                                // 3. Fallback
                                                return (
                                                     <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-900 text-zinc-700">
                                                        <Film size={24} />
                                                     </div>
                                                );
                                            })()}
    
                                            <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black via-black/80 to-transparent">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-[10px] font-mono text-neon-green bg-black/50 px-1 rounded border border-neon-green/30">
                                                        {Math.floor(frame.timestamp_seconds / 60)}:{(frame.timestamp_seconds % 60).toString().padStart(2, '0')}
                                                    </span>
                                                    <span className="text-[10px] text-zinc-400 truncate max-w-[100px]">
                                                        {frame.title}
                                                    </span>
                                                </div>
                                            </div>
                                            
                                            {/* Hover Play Button */}
                                            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40 backdrop-blur-[1px]">
                                                <div className="w-10 h-10 rounded-full bg-neon-green text-black flex items-center justify-center shadow-lg transform scale-90 group-hover:scale-100 transition-transform">
                                                    <Play size={20} fill="currentColor" />
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                 </div>
                             )}
                        </div>
                    )}
                 </div>

                 {/* GLOBAL VIDEO OVERLAY FOR CREATE TAB - Fixed Coverage */}
                 <AnimatePresence>
                  {pipFrame && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="fixed inset-0 z-[100] flex items-center justify-center p-6 bg-black/80 backdrop-blur-md"
                      onClick={() => setPipFrame(null)}
                    >
                      <div
                        className="relative w-full max-w-5xl aspect-video rounded-xl overflow-hidden border-2 border-neon-green/50 bg-black shadow-2xl shadow-neon-green/20"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          onClick={() => setPipFrame(null)}
                          className="absolute top-4 right-4 z-30 w-10 h-10 rounded-full bg-black/80 border border-zinc-600 flex items-center justify-center text-white hover:bg-red-500 hover:border-red-500 transition-colors"
                          aria-label="Close video"
                        >
                          <X size={20} />
                        </button>
                        <div className="absolute top-4 left-4 z-30 px-3 py-1.5 rounded bg-black/80 text-xs text-neon-green font-mono border border-neon-green/30 truncate max-w-[80%] flex items-center gap-2">
                          <Play size={12} fill="currentColor" />
                          <span className="font-bold">{pipFrame.title}</span> 
                          <span className="text-zinc-500">|</span> 
                          <span className="text-white">{pipFrame.timestamp_seconds != null ? `${Math.floor(pipFrame.timestamp_seconds / 60)}m ${pipFrame.timestamp_seconds % 60}s` : ""}</span>
                        </div>
                        
                        {embedUrlForSource(pipFrame.source_url, pipFrame.timestamp_seconds ?? 0) ? (
                          <iframe
                            src={(embedUrlForSource(pipFrame.source_url, pipFrame.timestamp_seconds ?? 0) ?? "") + "&autoplay=1"}
                            title={pipFrame.title}
                            className="w-full h-full"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                            allowFullScreen
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col items-center justify-center gap-4 text-zinc-400 p-8 bg-zinc-900/50">
                            <Video size={64} className="text-zinc-600" />
                            <div className="text-center space-y-1">
                                <h3 className="text-lg font-bold text-white">Playback Unavailable</h3>
                                <p className="text-sm">This source cannot be embedded directly.</p>
                            </div>
                            <a
                              href={pipFrame.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="bg-neon-green text-black px-6 py-2 rounded font-bold hover:bg-white hover:shadow-lg transition-all flex items-center gap-2"
                            >
                              Open in New Tab <Globe size={16} />
                            </a>
                            <button
                              type="button"
                              onClick={() => setPipFrame(null)}
                              className="text-zinc-500 hover:text-white text-xs underline underline-offset-4"
                            >
                              Close Overlay
                            </button>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                 </AnimatePresence>
             </div>
          ) : activeTab === "import" ? (
             <div className="h-full w-full p-8 overflow-y-auto">
                 <div className="max-w-4xl mx-auto space-y-8">
                    <div className="space-y-2">
                        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                            <Download className="text-neon-green" /> Import Dataset from CSV
                        </h1>
                        <p className="text-zinc-500">
                            Upload a descriptive query, a CSV file containing video URLs to process specific sources and data.
                        </p>
                    </div>

                    <div className="glass-panel p-6 space-y-6">
                        <div className="space-y-2">
                             <h2 className="text-sm font-bold uppercase text-zinc-400">Dataset Description</h2>
                             <textarea 
                                value={importPrompt}
                                onChange={(e) => setImportPrompt(e.target.value)}
                                placeholder="Describe what you want to find in these videos..."
                                className="w-full bg-zinc-900 border border-zinc-800 rounded p-3 text-sm text-white focus:outline-none focus:border-neon-green transition-colors min-h-[100px]"
                            />
                        </div>

                        <div className="space-y-2 pt-4 border-t border-zinc-800">
                             <h2 className="text-sm font-bold uppercase text-zinc-400">Upload CSV</h2>
                             <label className="border-2 border-dashed border-zinc-800 rounded-lg p-8 flex flex-col items-center justify-center gap-2 hover:border-neon-green hover:bg-zinc-900/50 transition-all cursor-pointer group">
                                <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
                                <div className="p-3 bg-zinc-900 rounded-full group-hover:scale-110 transition-transform">
                                   <Database size={24} className="text-zinc-500 group-hover:text-neon-green" />
                                </div>
                                <span className="text-sm text-zinc-400 font-bold">Click to Upload CSV</span>
                                <span className="text-xs text-zinc-600">Must contain a 'url' column</span>
                             </label>
                             {importedUrls.length > 0 && (
                                <div className="mt-2 text-xs text-neon-green flex items-center gap-2">
                                    <Shield size={12} /> Successfully loaded {importedUrls.length} valid URLs
                                </div>
                             )}
                        </div>

                        <div className="pt-4 border-t border-zinc-800 flex justify-end">
                            <button 
                                onClick={handleImportDataset} 
                                disabled={isImporting || importedUrls.length === 0}
                                className={cn(
                                    "px-8 py-3 rounded font-bold uppercase tracking-widest text-sm transition-all flex items-center gap-2",
                                    isImporting || importedUrls.length === 0
                                      ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                                      : "bg-neon-green text-black hover:bg-white hover:shadow-[0_0_20px_rgba(34,197,94,0.4)]"
                                )}
                            >
                                {isImporting ? <Activity className="animate-spin" /> : <Play size={18} fill="currentColor" />}
                                {isImporting ? "Processing Import..." : "Start Import Job"}
                            </button>
                        </div>
                    </div>
                 </div>
                 
                  {/* Reuse Dataset Visual Explorer for Import Results */}
                    {(datasetFrames.length > 0 || (isMonitoringDataset && importedUrls.length > 0)) && (
                         <div className="mt-8 max-w-4xl mx-auto space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-150">
                             <div className="flex justify-between items-center">
                                <h2 className="text-sm font-bold uppercase text-zinc-400 flex items-center">
                                    <Database className="inline mr-2 text-neon-green" size={16}/> Imported Clips Explorer
                                </h2>
                                <div className="flex items-center gap-2">
                                   {isMonitoringDataset && (
                                     <span className="flex items-center gap-2 text-[10px] text-neon-green animate-pulse">
                                        <Activity size={12} /> PROCESSING
                                     </span>
                                   )}
                                   <span className="text-xs bg-zinc-800 px-2 py-0.5 rounded text-zinc-400 font-mono">
                                       {datasetFrames.length} CLIPS PROCESSED
                                   </span>
                                   {importedUrls.length > 0 && (
                                     <span className="text-xs bg-zinc-800 px-2 py-0.5 rounded text-zinc-500 font-mono border border-zinc-700">
                                         FROM {importedUrls.length} VIDEOS
                                     </span>
                                   )}
                                </div>
                             </div>
                             
                             <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                                {datasetFrames.map((frame, i) => (
                                    <div 
                                        key={i} 
                                        className="group relative aspect-video bg-black rounded-lg border border-zinc-800 overflow-hidden hover:border-neon-green transition-all cursor-pointer shadow-lg hover:shadow-neon-green/10"
                                        onClick={() => setPipFrame({
                                            source_url: frame.source_url,
                                            timestamp_seconds: frame.timestamp_seconds,
                                            title: frame.title
                                        })}
                                    >
                                        <div className="absolute inset-0 flex items-center justify-center bg-zinc-900 text-zinc-700">
                                            <Film size={24} />
                                        </div>
                                        <img 
                                            src={frame.url || (frame.source_url.includes('youtube') ? `https://img.youtube.com/vi/${frame.source_url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/)?.[1]}/mqdefault.jpg` : '')} 
                                            className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity"
                                            onError={(e) => e.currentTarget.style.display = 'none'}
                                            alt=""
                                        />
                                        
                                        <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black via-black/80 to-transparent">
                                            <div className="flex items-center justify-between">
                                                <span className="text-[10px] font-mono text-neon-green bg-black/50 px-1 rounded border border-neon-green/30">
                                                    {Math.floor(frame.timestamp_seconds / 60)}:{(frame.timestamp_seconds % 60).toString().padStart(2, '0')}
                                                </span>
                                                <span className="text-[10px] text-zinc-400 truncate max-w-[100px]">
                                                    {frame.title}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                             </div>
                        </div>
                    )}

                 {/* GLOBAL VIDEO OVERLAY FOR IMPORT TAB */}
                 <AnimatePresence>
                  {pipFrame && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="fixed inset-0 z-[100] flex items-center justify-center p-6 bg-black/80 backdrop-blur-md"
                      onClick={() => setPipFrame(null)}
                    >
                      <div
                        className="relative w-full max-w-5xl aspect-video rounded-xl overflow-hidden border-2 border-neon-green/50 bg-black shadow-2xl shadow-neon-green/20"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          onClick={() => setPipFrame(null)}
                          className="absolute top-4 right-4 z-30 w-10 h-10 rounded-full bg-black/80 border border-zinc-600 flex items-center justify-center text-white hover:bg-red-500 hover:border-red-500 transition-colors"
                          aria-label="Close video"
                        >
                          <X size={20} />
                        </button>
                        <div className="absolute top-4 left-4 z-30 px-3 py-1.5 rounded bg-black/80 text-xs text-neon-green font-mono border border-neon-green/30 truncate max-w-[80%] flex items-center gap-2">
                          <Play size={12} fill="currentColor" />
                          <span className="font-bold">{pipFrame.title}</span> 
                          <span className="text-zinc-500">|</span> 
                          <span className="text-white">{pipFrame.timestamp_seconds != null ? `${Math.floor(pipFrame.timestamp_seconds / 60)}m ${pipFrame.timestamp_seconds % 60}s` : ""}</span>
                        </div>
                        
                        {embedUrlForSource(pipFrame.source_url, pipFrame.timestamp_seconds ?? 0) ? (
                          <iframe
                            src={(embedUrlForSource(pipFrame.source_url, pipFrame.timestamp_seconds ?? 0) ?? "") + "&autoplay=1"}
                            title={pipFrame.title}
                            className="w-full h-full"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                            allowFullScreen
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col items-center justify-center gap-4 text-zinc-400 p-8 bg-zinc-900/50">
                            <Video size={64} className="text-zinc-600" />
                            <div className="text-center space-y-1">
                                <h3 className="text-lg font-bold text-white">Playback Unavailable</h3>
                                <p className="text-sm">This source cannot be embedded directly.</p>
                            </div>
                            <a
                              href={pipFrame.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="bg-neon-green text-black px-6 py-2 rounded font-bold hover:bg-white hover:shadow-lg transition-all flex items-center gap-2"
                            >
                              Open in New Tab <Globe size={16} />
                            </a>
                            <button
                              type="button"
                              onClick={() => setPipFrame(null)}
                              className="text-zinc-500 hover:text-white text-xs underline underline-offset-4"
                            >
                              Close Overlay
                            </button>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                 </AnimatePresence>
             </div>
          ) : (
        <div className="flex h-full w-full p-4 gap-4">
      
      {/* --- LEFT COLUMN: CONTROL & TERMINAL --- */}
      <div className="w-1/4 flex flex-col gap-4">
        
        {/* HEADER REMOVED FROM HERE */}

        {/* REQUEST TERMINAL */}
        <div className="glass-panel p-6 flex-1 flex flex-col gap-6">
          <div className="flex items-center gap-2 text-neon-green mb-2">
            <Terminal size={18} />
            <h2 className="text-sm font-bold uppercase tracking-wider">Request Terminal</h2>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs text-zinc-500 uppercase font-bold">Detection Target</label>
              <div className="relative">
                <Search className="absolute left-3 top-2.5 text-zinc-500" size={16} />
                <input 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g. 'Red Bull Cans', 'Specific Dance'"
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 pl-10 pr-4 text-sm text-white focus:outline-none focus:border-neon-green transition-colors"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                 <label className="text-xs text-zinc-500 uppercase font-bold">Target Video URL</label>
                 {/* Source Indicator */}
                 {url && source !== "Unknown" ? (
                    <span className="text-[10px] text-neon-green bg-neon-green/10 border border-neon-green/20 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider fade-in">
                       {source} DETECTED
                    </span>
                 ) : url && (
                    <span className="text-[10px] text-red-400 bg-red-900/20 border border-red-900/50 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider fade-in">
                       UNSUPPORTED
                    </span>
                 )}
              </div>
              <div className="relative">
                <Video className="absolute left-3 top-2.5 text-zinc-500" size={16} />
                <input 
                  value={url}
                  onChange={(e) => {
                    const val = e.target.value;
                    setUrl(val);
                    
                    // Auto-detect Source
                    const isSupported = (
                        val.includes("youtube.com") || val.includes("youtu.be") || 
                        val.includes("twitch.tv") || val.includes("tiktok.com") ||
                        val.includes("vimeo.com") || val.includes("dailymotion.com") ||
                        val.includes("facebook.com")
                    );
                    
                    if (val.includes("youtube.com") || val.includes("youtu.be")) setSource("Youtube");
                    else if (val.includes("twitch.tv")) setSource("Twitch");
                    else if (val.includes("tiktok.com")) setSource("TikTok");
                    else if (isSupported) setSource("Generic");
                    else setSource("Unknown");
                  }}
                  placeholder="e.g. YouTube / Twitch URL"
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 pl-10 pr-4 text-sm text-white focus:outline-none focus:border-neon-green transition-colors"
                />
              </div>
            </div>

            <div className="space-y-2 pt-4 border-t border-zinc-800">
                <div className="flex justify-between items-end">
                    <label className="text-xs text-zinc-500 uppercase font-bold">Cluster Provisioning</label>
                    <div className="flex flex-col items-end">
                        <span className="text-sm font-mono font-bold text-white">{scale} <span className="text-zinc-500 text-[10px]">CONTAINERS</span></span>
                    </div>
                </div>

                {/* Visual Segmented Bar */}
                <div className="relative h-6 bg-zinc-900 rounded border border-zinc-800 flex overflow-hidden">
                     {/* GPU Segment (1-20) */}
                     <div 
                        className="h-full bg-neon-green/20 border-r border-zinc-800 flex items-center justify-center text-[9px] font-bold text-neon-green select-none transition-all"
                        style={{ width: `${Math.min(scale, 20)}%` }}
                     >
                        {scale > 0 && "GPU"}
                     </div>
                     
                     {/* CPU/Queued Segment (21-100) */}
                     <div 
                        className="h-full bg-neon-amber/20 flex items-center justify-center text-[9px] font-bold text-neon-amber select-none transition-all"
                        style={{ width: `${Math.max(0, scale - 20)}%` }}
                     >
                        {scale > 20 && "CPU"}
                     </div>

                     {/* Tick Marks Overlay */}
                     <div className="absolute inset-0 flex justify-between px-1 pointer-events-none">
                        {Array.from({length: 19}).map((_, i) => (
                           <div key={i} className={cn("w-[1px] h-full", (i+1) % 5 === 0 ? "bg-black/50" : "bg-transparent")} />
                        ))}
                     </div>
                </div>

                <input 
                    type="range" min="1" max="100" value={scale} 
                    onChange={(e) => setScale(parseInt(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-neon-green"
                />
                
                <div className="flex justify-between text-[10px] text-zinc-500 font-mono">
                     <span>1</span>
                     <span className="text-neon-green">20 A10G</span>
                     <span className="text-neon-amber">100 MAX</span>
                </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-zinc-800">
              <div className="flex items-center gap-2">
                <Shield size={14} className={stealth ? "text-neon-amber" : "text-zinc-600"} />
                <span className="text-xs font-bold text-zinc-400">STEALTH MODE</span>
              </div>
              <button 
                onClick={() => setStealth(!stealth)}
                className={cn(
                  "w-10 h-5 rounded-full relative transition-colors",
                  stealth ? "bg-neon-amber" : "bg-zinc-800"
                )}
              >
                <div className={cn(
                  "absolute top-1 w-3 h-3 bg-white rounded-full transition-all",
                  stealth ? "left-6" : "left-1"
                )} />
              </button>
            </div>
          </div>

          <div className="mt-auto space-y-4">
             {/* Progress Bar */}
             {(isIngesting || ingestionProgress.status === 'processing' || ingestionProgress.percent > 0) && (
                  <div className="space-y-1 animate-in fade-in slide-in-from-bottom-2">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-zinc-500 font-mono">
                          <span>Ingestion Progress</span>
                          <span className="text-neon-green">{Math.round(ingestionProgress.percent)}% ({ingestionProgress.current}/{ingestionProgress.total})</span>
                      </div>
                      <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden border border-zinc-800">
                          <div 
                              className="h-full bg-neon-green shadow-[0_0_10px_currentColor] transition-all duration-500 ease-out relative"
                              style={{ width: `${Math.max(2, ingestionProgress.percent)}%` }}
                          >
                            <div className="absolute inset-0 bg-white/20 animate-pulse" />
                          </div>
                      </div>
                  </div>
             )}

             <button
                onClick={isIngesting ? handleStop : handleStart}
                className={cn(
                  "w-full py-4 rounded-lg font-bold text-sm tracking-widest uppercase transition-all flex items-center justify-center gap-2 shadow-lg",
                  isIngesting 
                    ? "bg-red-900/50 border border-red-500 text-red-100 hover:bg-red-900" 
                    : "bg-neon-green/10 border border-neon-green text-neon-green hover:bg-neon-green/20"
                )}
             >
                {isIngesting ? <><Pause size={16} /> ABORT SEQUENCE</> : <><Play size={16} /> INITIALIZE INGEST</>}
             </button>
          </div>
        </div>
      </div>

      {/* --- CENTER COLUMN: LIVE EXTRACTION --- */}
      <div className="flex-1 flex flex-col gap-4">
        
        {/* METRICS ROW */}
        <div className="grid grid-cols-3 gap-4 h-32">
          {[
            { label: "Active Workers", val: stats.active_workers, unit: "NODES", icon: Cpu, color: "text-blue-400" },
            { label: "Throughput", val: stats.fps, unit: "FPS", icon: Zap, color: "text-neon-green" },
            { label: "Bandwidth", val: stats.bandwidth, unit: "MB/s", icon: Activity, color: "text-neon-amber" },
          ].map((m, i) => (
            <div key={i} className="glass-panel p-4 flex flex-col justify-between relative overflow-hidden group">
               <div className="flex justify-between items-start z-10">
                  <span className="text-xs text-zinc-500 uppercase font-bold">{m.label}</span>
                  <m.icon size={16} className={m.color} />
               </div>
               <div className="z-10">
                  <span className={cn("text-3xl font-mono font-bold", m.color)}>{m.val}</span>
                  <span className="text-xs text-zinc-600 ml-2 font-bold">{m.unit}</span>
               </div>
               {/* Background Chart Effect */}
               <div className={cn("absolute bottom-0 left-0 right-0 h-16 opacity-10 group-hover:opacity-20 transition-opacity", m.color)}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <Line type="monotone" dataKey="fps" stroke="currentColor" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
               </div>
            </div>
          ))}
        </div>

        {/* WORKER VISUALIZATION */}
        <div className="glass-panel flex-1 p-4 relative overflow-hidden flex flex-col h-[500px]">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-zinc-400">
                <Layers size={16} />
                <span className="text-xs font-bold uppercase">Infrastructure Mesh</span>
              </div>
              <div className="flex gap-2">
                <div className="flex items-center gap-1 text-[10px] text-zinc-500"><div className="w-2 h-2 bg-neon-green rounded-sm" /> A10G</div>
                <div className="flex items-center gap-1 text-[10px] text-zinc-500"><div className="w-2 h-2 bg-neon-amber rounded-sm" /> CPU</div>
              </div>
            </div>
            
            <div className="grid grid-cols-10 gap-1.5 content-start overflow-hidden relative">
              {/* Scanline Effect */}
              {isIngesting && (
                 <motion.div 
                    initial={{ top: "-10%" }}
                    animate={{ top: "110%" }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                    className="absolute left-0 right-0 h-20 bg-gradient-to-b from-transparent via-neon-green/10 to-transparent z-10 pointer-events-none"
                 />
              )}

              {Array.from({ length: 100 }).map((_, i) => {
                const isActive = i < scale;
                const isGPU = i < 20;

                // Determine colors based on state
                let borderColor = "#18181b"; // default dark
                let bgColor = "#09090b"; // default bg

                if (isActive) {
                    if (isIngesting) {
                        // ACTIVE INGESTION
                        borderColor = isGPU ? "#22c55e" : "#f59e0b"; // Neon Green : Amber
                        bgColor = isGPU ? "rgba(34, 197, 94, 0.2)" : "rgba(245, 158, 11, 0.2)";
                    } else {
                        // PROVISIONED (IDLE)
                        borderColor = isGPU ? "#15803d" : "#b45309"; // Darker Green : Darker Amber
                        bgColor = isGPU ? "rgba(21, 128, 61, 0.1)" : "rgba(180, 83, 9, 0.1)"; 
                    }
                } else {
                    // UNALLOCATED
                    borderColor = "#27272a"; 
                    bgColor = "#09090b";
                }
                
                return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0.2, scale: 0.8 }}
                  animate={{ 
                    opacity: isActive ? 1 : 0.3,
                    scale: isActive ? 1 : 0.85,
                    borderColor: borderColor,
                    backgroundColor: bgColor
                  }}
                  transition={{ duration: 0.3, delay: i * 0.002 }}
                  className={cn(
                    "aspect-square rounded-[2px] border flex items-center justify-center relative group transition-colors",
                  )}
                >
                   {isActive && isIngesting && (
                     <motion.div 
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ repeat: Infinity, duration: 1 + Math.random() }}
                        className={cn("w-1.5 h-1.5 rounded-full shadow-[0_0_5px_currentColor]", isGPU ? "bg-neon-green text-neon-green" : "bg-neon-amber text-neon-amber")}
                     />
                   )}
                   
                   {/* Tooltip */}
                   <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-30 whitespace-nowrap bg-black border border-zinc-800 px-2 py-1 text-[10px] text-white rounded pointer-events-none shadow-xl">
                      <div className="font-bold mb-0.5">{isGPU ? "NVIDIA A10G Tensor Core" : "CPU Worker Node"}</div>
                      <div className="text-zinc-500 text-[9px]">{isActive ? (isIngesting ? "STATUS: PROCESSING" : "STATUS: STANDBY") : "STATUS: OFFLINE"}</div>
                   </div>
                </motion.div>
              )})}
            </div>

            {/* PiP video overlay over cluster — starts at frame timestamp */}
            <AnimatePresence>
              {pipFrame && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="fixed inset-0 z-[200] flex items-center justify-center p-6 bg-black/60 backdrop-blur-sm"
                  onClick={() => setPipFrame(null)}
                >
                  <div
                    className="relative w-full max-w-2xl aspect-video rounded-xl overflow-hidden border-2 border-neon-green/50 bg-black shadow-2xl shadow-neon-green/20"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      type="button"
                      onClick={() => setPipFrame(null)}
                      className="absolute top-2 right-2 z-30 w-8 h-8 rounded-full bg-black/80 border border-zinc-600 flex items-center justify-center text-white hover:bg-red-500 hover:border-red-500 transition-colors"
                      aria-label="Close video"
                    >
                      <X size={16} />
                    </button>
                    <div className="absolute top-2 left-2 z-30 px-2 py-1 rounded bg-black/80 text-[10px] text-neon-green font-mono border border-neon-green/30 truncate max-w-[80%]">
                      {pipFrame.title} · {pipFrame.timestamp_seconds != null ? `${Math.floor(pipFrame.timestamp_seconds / 60)}m ${pipFrame.timestamp_seconds % 60}s` : ""}
                    </div>
                    {embedUrlForSource(pipFrame.source_url, pipFrame.timestamp_seconds ?? 0) ? (
                      <iframe
                        src={embedUrlForSource(pipFrame.source_url, pipFrame.timestamp_seconds ?? 0) ?? ""}
                        title={pipFrame.title}
                        className="w-full h-full"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                        allowFullScreen
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-zinc-400 p-4">
                        <Video size={48} className="text-zinc-600" />
                        <span className="text-sm text-center">This source cannot be embedded. Open in a new tab to watch.</span>
                        <a
                          href={pipFrame.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-neon-green hover:underline text-sm font-bold"
                        >
                          Open video →
                        </a>
                        <button
                          type="button"
                          onClick={() => setPipFrame(null)}
                          className="mt-2 text-zinc-500 hover:text-white text-xs"
                        >
                          Close
                        </button>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            
            {/* TERMINAL LOGS OVERLAY */}
            <div 
              ref={termRef}
              style={{ height: isTermExpanded ? 'calc(100vh - 300px)' : `${termHeight}px`, minHeight: '192px', maxHeight: '800px' }}
              className="mt-4 bg-zinc-950 border border-zinc-800 rounded-lg flex flex-col font-mono text-xs overflow-hidden transition-all relative shadow-2xl"
            >
               {/* Drag Handle */}
               <div 
                 className="h-1 bg-zinc-800 hover:bg-neon-green cursor-ns-resize w-full flex justify-center py-[2px] mb-1 opacity-50 hover:opacity-100 transition-opacity"
                 onMouseDown={(e) => {
                    const startY = e.clientY;
                    const startH = termHeight;
                    const onMove = (moveEvent: any) => {
                       const newH = startH - (moveEvent.clientY - startY);
                       setTermHeight(Math.max(192, Math.min(newH, 600)));
                    };
                    const onUp = () => {
                       document.removeEventListener('mousemove', onMove);
                       document.removeEventListener('mouseup', onUp);
                    };
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                 }}
               >
                  <div className="w-8 h-full bg-zinc-600 rounded-full" />
               </div>

               {/* Toolbar */}
               <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-900 bg-zinc-950/90 backdrop-blur sticky top-0 z-20">
                  <div className="flex items-center gap-2 text-zinc-500">
                     <Terminal size={14} className="text-neon-green" />
                     <span className="font-bold">root@ingest-ai:~# tail -f /var/log/ingest.log</span>
                  </div>
                  <div className="flex items-center gap-2">
                     <div className="relative group">
                        <Search size={14} className="absolute left-2 top-1.5 text-zinc-500" />
                        <input 
                           value={termSearch}
                           onChange={(e) => setTermSearch(e.target.value)}
                           className="bg-zinc-900 border border-zinc-800 rounded px-2 pl-7 py-1 w-32 focus:w-64 transition-all outline-none text-zinc-300 placeholder:text-zinc-600"
                           placeholder="grep logs..."
                        />
                     </div>
                     <button 
                        onClick={() => navigator.clipboard.writeText(logs.join('\n'))}
                        className="p-1.5 hover:bg-zinc-800 rounded text-zinc-500 hover:text-white transition-colors"
                        title="Copy to Clipboard"
                     >
                        <Download size={14} className="rotate-180" /> {/* Upload icon used as copy */}
                     </button>
                     <button 
                        onClick={() => setIsTermExpanded(!isTermExpanded)}
                        className="p-1.5 hover:bg-zinc-800 rounded text-zinc-500 hover:text-white transition-colors"
                     >
                        {isTermExpanded ? <div className="w-3 h-1 bg-current" /> : <div className="w-3 h-3 border-2 border-current" />}
                     </button>
                  </div>
               </div>

               {/* Logs Body */}
               <div className="flex-1 overflow-y-auto p-3 space-y-1 custom-scrollbar scroll-smooth">
                 <AnimatePresence>
                   {(termSearch ? logs.filter(l => l.toLowerCase().includes(termSearch.toLowerCase())) : logs).map((log, i) => (
                     <motion.div 
                        initial={{ opacity: 0, x: -10 }} 
                        animate={{ opacity: 1, x: 0 }}
                        key={i} className="text-zinc-400 break-words whitespace-pre-wrap"
                      >
                       <span className="text-zinc-600 mr-2 select-none">{log.split('] ')[0]}]</span> 
                       <span className={log.includes("100%") || log.includes("SUCCESS") ? "text-neon-green" : log.includes("ERROR") ? "text-red-400" : "text-zinc-300"}>
                         {log.split('] ')[1] || log}
                       </span>
                     </motion.div>
                   ))}
                  </AnimatePresence>
                  <div ref={logsEndRef} />
               </div>
            </div>
        </div>
      </div>

      {/* --- RIGHT COLUMN: DATASET EXPLORER --- */}
      <div className="w-1/4 glass-panel flex flex-col overflow-hidden">
        <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
            <div className="flex items-center gap-2 text-zinc-400">
              <Database size={16} />
              <span className="text-xs font-bold uppercase">Dataset Explorer</span>
            </div>
            <span className="text-[10px] bg-zinc-800 px-2 py-0.5 rounded text-white">{frames.length} items</span>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
           {frames.length === 0 ? (
             <div className="h-full flex flex-col items-center justify-center text-zinc-600 gap-2">
                <AlertCircle size={32} />
                <span className="text-xs">No samples yet. Start ingestion.</span>
             </div>
           ) : (
             <div className="grid grid-cols-1 gap-3">
               {frames.map((frame, i) => (
                 <motion.div
                    role="button"
                    tabIndex={0}
                    onClick={() => setPipFrame({
                      source_url: frame.source_url,
                      timestamp_seconds: frame.timestamp_seconds ?? 0,
                      title: frame.title ?? "Video",
                    })}
                    onKeyDown={(e) => e.key === "Enter" && setPipFrame({
                      source_url: frame.source_url,
                      timestamp_seconds: frame.timestamp_seconds ?? 0,
                      title: frame.title ?? "Video",
                    })}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={i}
                    className="group relative aspect-video bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden hover:border-neon-green transition-all cursor-pointer"
                  >
                   {/* Twitch Logo Overlay */}
                    {frame.source_url && frame.source_url.includes('twitch') ? (
                         <div className="w-full h-full bg-[#9146FF]/20 flex items-center justify-center opacity-60 group-hover:opacity-100 transition-opacity">
                            <Twitch size={48} className="text-[#9146FF]" />
                         </div>
                    ) : (
                        <img src={frame.url} className="w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity" alt="" />
                    )}
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="w-10 h-10 bg-neon-green/90 rounded-full flex items-center justify-center shadow-lg shadow-neon-green/20 backdrop-blur-sm">
                            <Play size={18} className="text-black ml-1" fill="currentColor" />
                        </div>
                    </div>
                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black via-black/90 to-transparent p-3 pt-8">
                       <div className="flex justify-between items-end">
                          <div className="flex flex-col gap-0.5">
                             {/* Twitch Styling Fix */}
                             <div className="flex items-center gap-1.5 text-[10px] text-neon-green font-mono font-bold">
                                {frame.source_url && frame.source_url.includes('twitch') ? (
                                    <Twitch size={10} />
                                ) : (
                                    <Activity size={10} />
                                )}
                                {(parseFloat(frame.score) * 100).toFixed(1)}% MATCH
                             </div>
                             <div className="text-xs text-white font-bold truncate max-w-[120px]">{frame.title}</div>
                          </div>
                          <span className="text-[10px] bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded border border-zinc-700 font-mono">
                            {/* @ts-ignore */}
                            {frame.timestamp_display || frame.timestamp}
                          </span>
                       </div>
                    </div>
                 </motion.div>
               ))}
             </div>
           )}
        </div>
        
        <div className="p-4 border-t border-zinc-800 bg-zinc-950/50">
           <button className="w-full py-2 bg-zinc-100 text-zinc-900 font-bold text-xs rounded hover:bg-white flex items-center justify-center gap-2">
             <Download size={14} /> EXPORT DATASET (JSON)
           </button>
        </div>
      </div>
      
      </div>
        )}
        </div>
    </div>
  );
}
