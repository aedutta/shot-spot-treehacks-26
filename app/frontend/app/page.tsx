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
  const [source, setSource] = useState("Youtube");
  const [scale, setScale] = useState(10);
  const [stealth, setStealth] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  
  const [stats, setStats] = useState({ active_workers: 0, fps: 0, bandwidth: 0 });
  const [chartData, setChartData] = useState(generateChartData());
  const [logs, setLogs] = useState<string[]>([]);
  const [frames, setFrames] = useState<any[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // --- Effects ---
  
  // Real-time chart updates
  useEffect(() => {
    if (!isIngesting) return;
    const interval = setInterval(() => {
      setChartData(prev => {
        const newData = [...prev.slice(1), {
          time: prev[prev.length - 1].time + 1,
          fps: Math.floor(Math.random() * 100) + (scale * 20),
          bandwidth: Math.floor(Math.random() * 50) + (scale * 5),
        }];
        return newData;
      });
      
      // Update stats based on "simulated" backend
      setStats({
        active_workers: Math.min(scale, Math.floor(Math.random() * scale) + (scale/2)),
        fps: Math.floor(Math.random() * 100) + (scale * 20),
        bandwidth: Math.floor(Math.random() * 50) + (scale * 5),
      });

      // Add Random Log
      if (Math.random() > 0.7) {
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
    addLog(`INITIALIZING CLUSTER...`);
    addLog(`Source: ${source} | Target: "${prompt}"`);
    addLog(`Scaling to ${scale} workers on A10G GPUs...`);
    if (stealth) addLog(`[STEALTH] Bright Data Resi-Proxies ENGAGED 🥷`);
    
    // Simulate finding frames
    setTimeout(() => {
      fetchFrames();
    }, 3000);
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
      const res = await axios.post("http://localhost:8000/search", { query: prompt, top_k: 12 });
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
              <label className="text-xs text-zinc-500 uppercase font-bold">Data Source</label>
              <div className="grid grid-cols-3 gap-2">
                {["Youtube", "Twitch", "TikTok"].map((s) => (
                  <button
                    key={s}
                    onClick={() => setSource(s)}
                    className={cn(
                      "py-2 text-xs font-bold border rounded-md transition-all",
                      source === s 
                        ? "bg-zinc-800 border-neon-green text-white" 
                        : "bg-zinc-900 border-zinc-800 text-zinc-500 hover:bg-zinc-800"
                    )}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4 pt-4 border-t border-zinc-800">
              <div className="flex justify-between">
                <label className="text-xs text-zinc-500 uppercase font-bold">Concurrency (Workers)</label>
                <span className="text-xs font-mono text-neon-green">{scale}x A10G</span>
              </div>
              <input 
                type="range" min="1" max="100" value={scale} 
                onChange={(e) => setScale(parseInt(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-neon-green"
              />
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
        <div className="glass-panel flex-1 p-4 relative overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-zinc-400">
                <Layers size={16} />
                <span className="text-xs font-bold uppercase">Infrastructure Mesh</span>
              </div>
              <div className="flex gap-2">
                <div className="flex items-center gap-1 text-[10px] text-zinc-500"><div className="w-2 h-2 bg-neon-green rounded-full" /> PROVISIONED</div>
                <div className="flex items-center gap-1 text-[10px] text-zinc-500"><div className="w-2 h-2 bg-zinc-800 rounded-full" /> IDLE</div>
              </div>
            </div>
            
            <div className="grid grid-cols-10 gap-1 content-start overflow-y-auto pr-2 custom-scrollbar">
              {Array.from({ length: 100 }).map((_, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0.3, scale: 0.8 }}
                  animate={{ 
                    opacity: i < scale ? 1 : 0.2,
                    scale: i < scale ? 1 : 0.9,
                    backgroundColor: i < scale ? (isIngesting ? "#22c55e" : "#f59e0b") : "#18181b"
                  }}
                  transition={{ duration: 0.2, delay: i * 0.005 }}
                  className="aspect-square rounded-[2px] border border-black/20"
                />
              ))}
            </div>
            
            {/* TERMINAL LOGS OVERLAY */}
            <div className="mt-4 h-48 bg-zinc-950 border border-zinc-800 rounded-lg p-3 font-mono text-xs overflow-y-auto">
               <div className="text-zinc-500 mb-2 sticky top-0 bg-zinc-950 pb-2 border-b border-zinc-900">
                  root@ingest-ai:~# tail -f /var/log/ingest.log
               </div>
               <div className="space-y-1">
                 <AnimatePresence>
                   {logs.map((log, i) => (
                     <motion.div 
                        initial={{ opacity: 0, x: -10 }} 
                        animate={{ opacity: 1, x: 0 }}
                        key={i} className="text-zinc-400"
                      >
                       <span className="text-zinc-600 mr-2">{log.split(']')[0]}]</span> 
                       <span className={log.includes("100%") ? "text-neon-green" : "text-zinc-300"}>
                         {log.split(']')[1]}
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
             <div className="grid grid-cols-2 gap-3">
               {frames.map((frame, i) => (
                 <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={i} 
                    className="group relative aspect-video bg-zinc-900 border border-zinc-800 rounded overflow-hidden hover:border-neon-green transition-colors cursor-pointer"
                  >
                    <img src={frame.url} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" />
                    
                    <div className="absolute inset-x-0 bottom-0 bg-black/80 backdrop-blur-sm p-2 transform translate-y-full group-hover:translate-y-0 transition-transform">
                       <div className="flex justify-between items-center text-[10px]">
                          <span className="text-neon-green font-mono">{(parseFloat(frame.score) * 100).toFixed(1)}%</span>
                          <span className="text-zinc-400">{frame.timestamp}</span>
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
  );
}
