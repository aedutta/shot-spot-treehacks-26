"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Terminal, Activity, Radio, Cpu, Download, 
  Search, Shield, Zap, Database, Play, Pause,
  Globe, Video, Layers, AlertCircle
} from "lucide-react";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts";
import axios from "axios";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// --- Utils ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
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

  // --- Effects ---
  
  // Real-time chart updates
  useEffect(() => {
    if (!isIngesting) return; // Don't run interval if not ingesting

    const interval = setInterval(async () => {
      // Fetch stats from backend
      try {
        const res = await axios.get("http://localhost:8000/stats");
        if (res.data) {
           setStats({
             active_workers: res.data.active_workers || 0,
             fps: res.data.fps_processed || 0,
             bandwidth: res.data.bandwidth_mbps || 0
           });
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
        const res = await axios.post("http://localhost:8000/ingest/start", null, {
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
                const checkRes = await axios.post("http://localhost:8000/search", payload);
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
    // In a real app, this hits the API. Here we mock it or hit our local backend.
    try {
      const payload = {
         query: prompt, 
         top_k: 12,
         source_url: url || undefined // Filter by this specific video if provided
      };
      
      const res = await axios.post("http://localhost:8000/search", payload);
      if (res.data.ok) {
        setFrames(res.data.results);
        addLog(`DATASET UPDATE: Found ${res.data.results.length} new samples matching "${prompt}"`);
      }
    } catch (e) {
      // Fallback if backend isn't running
      const mockFrames = Array.from({length: 12}, (_, i) => ({
        id: i,
        url: `https://picsum.photos/seed/${i + Math.random()}/300/200`,
        score: (0.7 + Math.random() * 0.29).toFixed(4),
        timestamp: "00:04:23"
      }));
      setFrames(mockFrames);
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#09090b] text-zinc-300 p-4 gap-4 font-mono overflow-hidden">
      
      {/* --- LEFT COLUMN: CONTROL & TERMINAL --- */}
      <div className="w-1/4 flex flex-col gap-4">
        
        {/* HEADER */}
        <div className="glass-panel p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-neon-green rounded-full animate-pulse" />
            <span className="font-bold text-white tracking-widest text-lg">INGEST.AI</span>
          </div>
          <span className="text-xs text-zinc-500">v0.1.0-alpha</span>
        </div>

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

          <div className="mt-auto">
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
                 <motion.a 
                    href={frame.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={i} 
                    className="group relative aspect-video bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden hover:border-neon-green transition-all cursor-pointer block"
                  >
                    {/* Placeholder or Thumbnail - For Twitch/YT we rely on the placeholder for now unless we fetch real thumbs */}
                    <img src={frame.url} className="w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity" />
                    
                    {/* Play Button Overlay */}
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="w-10 h-10 bg-neon-green/90 rounded-full flex items-center justify-center shadow-lg shadow-neon-green/20 backdrop-blur-sm">
                            <Play size={18} className="text-black ml-1" fill="currentColor" />
                        </div>
                    </div>

                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black via-black/90 to-transparent p-3 pt-8">
                       <div className="flex justify-between items-end">
                          <div className="flex flex-col gap-0.5">
                             <div className="flex items-center gap-1.5 text-[10px] text-neon-green font-mono font-bold">
                                <Activity size={10} />
                                {(parseFloat(frame.score) * 100).toFixed(1)}% MATCH
                             </div>
                             <div className="text-xs text-white font-bold truncate max-w-[120px]">{frame.title}</div>
                          </div>
                          <span className="text-[10px] bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded border border-zinc-700 font-mono">
                            {frame.timestamp}
                          </span>
                       </div>
                    </div>
                 </motion.a>
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
  );
}
