"use client";

import React, { useCallback, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Image as ImageIcon,
  Sparkles,
  Wand2,
  Settings2,
  Play,
  Trash2,
  Info,
  ChevronRight,
  Check,
  Loader2,
  Download,
  ShieldCheck,
  GitBranch,
  SlidersHorizontal,
  SquareCheckBig,
} from "lucide-react";

// shadcn/ui components
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Switch } from "../components/ui/switch";
import { Slider } from "../components/ui/slider";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../components/ui/tooltip";

// API helpers
import { removeBgFile } from "../lib/api";

// --- Utility types
interface PreviewItem {
  id: string;
  file: File;
  url: string;
  status: "queued" | "processing" | "done" | "error";
  cutoutUrl?: string;
  upscale?: number;
}

const gradientBg =
  "bg-[radial-gradient(80rem_40rem_at_20%_-10%,rgba(56,189,248,.25),transparent),radial-gradient(80rem_40rem_at_80%_10%,rgba(167,139,250,.25),transparent)]";

export default function BackgroundRemoverNextUI() {
  const [items, setItems] = useState<PreviewItem[]>([]);
  const [processing, setProcessing] = useState(false);

  const [model, setModel] = useState("u2net");
  const [enableUpscale, setEnableUpscale] = useState(true);
  const [scale, setScale] = useState(2);
  const [upscaleMethod, setUpscaleMethod] = useState("lanczos");
  const [whiteBg, setWhiteBg] = useState(false);
  const [padding, setPadding] = useState(10);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onSelectFiles = useCallback((files: FileList | null) => {
    if (!files || !files.length) return;
    const next: PreviewItem[] = Array.from(files).map((f) => ({
      id: `${f.name}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      file: f,
      url: URL.createObjectURL(f),
      status: "queued",
    }));
    setItems((prev) => [...prev, ...next]);
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    onSelectFiles(e.dataTransfer.files);
  }, [onSelectFiles]);

  const startProcessing = async () => {
    if (!items.length) return;
    setProcessing(true);
    
    // Process each item with real API calls
    for (const item of items) {
      if (item.status === "queued") {
        setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, status: "processing" } : p)));
        
        try {
          const result = await removeBgFile(item.file, {
            model,
            padding,
            white_bg: whiteBg,
            enhance: enableUpscale,
            response_type: "base64"
          });
          
          setItems((prev) =>
            prev.map((p) =>
              p.id === item.id
                ? {
                    ...p,
                    status: "done",
                    cutoutUrl: result.image,
                    upscale: enableUpscale ? scale : 1,
                  }
                : p,
            ),
          );
        } catch (error) {
          console.error("Processing error:", error);
          setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, status: "error" } : p)));
        }
      }
    }
    setProcessing(false);
  };

  const clearItem = (id: string) => setItems((prev) => prev.filter((p) => p.id !== id));

  const processedCount = useMemo(() => items.filter((i) => i.status === "done").length, [items]);

  return (
    <TooltipProvider>
      <div className={`min-h-screen ${gradientBg} transition-colors duration-300 bg-white`}>        
        {/* Top bar */}
        <header className="sticky top-0 z-30 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-black/40 border-b border-white/20 dark:border-white/10">
          <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-2xl bg-gradient-to-br from-sky-400 to-violet-500 grid place-items-center shadow-inner">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <div>
                <p className="text-sm text-neutral-500 dark:text-neutral-400 leading-none">AI Background Remover</p>
                <h1 className="text-lg font-semibold tracking-tight">Next‑Level UI Preview</h1>
              </div>
              <Badge variant="secondary" className="ml-2">Beta</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button 
                size="sm" 
                className="gap-2"
                onClick={() => window.open('https://github.com/kheloaki/ai-background-remover/blob/main/API_DOCUMENTATION.md', '_blank')}
              >
                <GitBranch className="h-4 w-4" /> API Docs
              </Button>
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="mx-auto max-w-7xl px-4 pt-10 pb-6">
          <div className="grid lg:grid-cols-12 gap-6 items-start">
            {/* Upload area */}
            <Card className="lg:col-span-8 border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur-xl shadow-xl">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" /> Upload Your Images
                </CardTitle>
                <CardDescription>
                  Drag & drop PNG/JPG/WebP here or click to browse. We process images with AI-powered background removal.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                  }}
                  onDrop={onDrop}
                  onClick={() => inputRef.current?.click()}
                  className="group relative grid place-items-center rounded-2xl border border-dashed border-neutral-300/70 dark:border-neutral-700/60 p-6 sm:p-10 min-h-[220px] cursor-pointer overflow-hidden"
                >
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={(e) => onSelectFiles(e.target.files)}
                  />
                  <motion.div
                    initial={{ scale: 0.96, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="text-center max-w-md"
                  >
                    <div className="mx-auto h-16 w-16 rounded-2xl bg-gradient-to-br from-sky-400 to-violet-500 grid place-items-center shadow-lg">
                      <ImageIcon className="h-8 w-8 text-white" />
                    </div>
                    <p className="mt-4 text-lg font-medium">Drop images to queue</p>
                    <p className="text-sm text-neutral-500 dark:text-neutral-400">High‑quality cutouts with smart edge refinement and optional upscaling.</p>
                    <div className="flex items-center justify-center gap-3 mt-5">
                      <Button className="gap-2"><Upload className="h-4 w-4"/>Choose Files</Button>
                      <Button variant="secondary" className="gap-2"><Info className="h-4 w-4"/>How it works</Button>
                    </div>
                  </motion.div>

                  <div className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="absolute -inset-40 bg-gradient-to-tr from-sky-400/10 via-fuchsia-400/10 to-violet-500/10 blur-3xl" />
                  </div>
                </div>

                {/* Queue */}
                <AnimatePresence initial={false}>
                  {items.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 8 }}
                      className="mt-6"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <p className="text-sm text-neutral-500 dark:text-neutral-400">
                          {processedCount}/{items.length} processed
                        </p>
                        <div className="flex items-center gap-2">
                          <Button variant="secondary" size="sm" onClick={() => setItems([])} disabled={processing} className="gap-2">
                            <Trash2 className="h-4 w-4"/> Clear
                          </Button>
                          <Button size="sm" onClick={startProcessing} disabled={processing || !items.length} className="gap-2">
                            {processing ? <Loader2 className="h-4 w-4 animate-spin"/> : <Play className="h-4 w-4"/>}
                            {processing ? "Processing…" : "Process Images"}
                          </Button>
                        </div>
                      </div>

                      <ul className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {items.map((it) => (
                          <li key={it.id} className="relative">
                            <div className="overflow-hidden rounded-2xl border border-white/20 dark:border-white/10 bg-white/60 dark:bg-white/5">
                              <div className="aspect-[4/3] w-full overflow-hidden">
                                <img src={it.url} alt={it.file.name} className="h-full w-full object-cover" />
                              </div>
                              <div className="p-3 flex items-center justify-between">
                                <div className="min-w-0">
                                  <p className="truncate text-sm font-medium">{it.file.name}</p>
                                  <div className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
                                    <SquareCheckBig className={`h-3.5 w-3.5 ${it.status === "done" ? "text-green-500" : it.status === "processing" ? "text-amber-500" : it.status === "error" ? "text-red-500" : "text-neutral-400"}`} />
                                    <span className="capitalize">{it.status}</span>
                                  </div>
                                </div>
                                <div className="flex items-center gap-1">
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button variant="ghost" size="icon" onClick={() => clearItem(it.id)}>
                                        <Trash2 className="h-4 w-4" />
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Remove from queue</TooltipContent>
                                  </Tooltip>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button variant="ghost" size="icon" disabled={it.status !== "done"}>
                                        <Download className="h-4 w-4" />
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Download cutout</TooltipContent>
                                  </Tooltip>
                                </div>
                              </div>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>

            {/* Controls */}
            <div className="lg:col-span-4 space-y-6">
              <Card className="border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur-xl shadow-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2"><Wand2 className="h-5 w-5"/> AI Model</CardTitle>
                  <CardDescription>Choose the segmentation model best suited for your images.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Label className="text-xs">Model Type</Label>
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="u2net">U²‑Net (General)</SelectItem>
                        <SelectItem value="u2netp">U²‑Net+ (Fast)</SelectItem>
                        <SelectItem value="u2net_human_seg">U²‑Net Human</SelectItem>
                        <SelectItem value="u2net_cloth_seg">U²‑Net Cloth</SelectItem>
                        <SelectItem value="isnet-general-use">ISNet General</SelectItem>
                        <SelectItem value="silueta">Silueta</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>

                  <div className="flex items-center justify-between pt-2">
                    <div className="space-y-0.5">
                      <Label className="text-xs">Edge Refinement</Label>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">Feather & preserve fine hair details.</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur-xl shadow-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2"><Settings2 className="h-5 w-5"/> Processing Options</CardTitle>
                  <CardDescription>Tweak output quality, speed, and resizing.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">Enable Upscaling</Label>
                    <Switch checked={enableUpscale} onCheckedChange={setEnableUpscale} />
                  </div>

                  <div className={`${enableUpscale ? "opacity-100" : "opacity-50"}`}>
                    <Label className="text-xs">Scale Factor: {scale.toFixed(1)}×</Label>
                    <Slider value={[scale]} min={1} max={4} step={0.1} onValueChange={(v) => setScale(v[0])} />
                  </div>

                  <div>
                    <Label className="text-xs">Upscale Method</Label>
                    <Select value={upscaleMethod} onValueChange={setUpscaleMethod}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a method" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="lanczos">Lanczos (Fast)</SelectItem>
                        <SelectItem value="bicubic">Bicubic</SelectItem>
                        <SelectItem value="ai">AI Enhanced</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur-xl shadow-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2"><ImageIcon className="h-5 w-5"/> Background Options</CardTitle>
                  <CardDescription>Set final canvas and padding for your exports.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">White Background</Label>
                    <Switch checked={whiteBg} onCheckedChange={setWhiteBg} />
                  </div>
                  <div>
                    <Label className="text-xs">Padding (pixels): {padding}</Label>
                    <Slider value={[padding]} min={0} max={80} step={2} onValueChange={(v) => setPadding(v[0])} />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <Button variant="secondary" className="gap-2"><ShieldCheck className="h-4 w-4"/> Safe Mode</Button>
                    <Button className="gap-2" onClick={startProcessing} disabled={processing || !items.length}>
                      {processing ? <Loader2 className="h-4 w-4 animate-spin"/> : <Play className="h-4 w-4"/>}
                      Process
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Mini FAQ / Tips */}
              <Card className="border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur-xl">
                <CardHeader className="pb-1">
                  <CardTitle className="text-base flex items-center gap-2"><Info className="h-5 w-5"/> Tips</CardTitle>
                </CardHeader>
                <CardContent className="text-sm space-y-2 text-neutral-600 dark:text-neutral-300">
                  <p><strong>⌘/Ctrl + U</strong> to open the uploader. <strong>⌘/Ctrl + Enter</strong> to process.</p>
                  <p>Portraits look best with <em>U²‑Net Human</em>. Products shine with <em>ISNet</em> or <em>U²‑Net</em>.</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* Results rail */}
        <section className="mx-auto max-w-7xl px-4 pb-16">
          <Tabs defaultValue="results" className="w-full">
            <TabsList className="grid grid-cols-2 max-w-md">
              <TabsTrigger value="results" className="gap-2"><Check className="h-4 w-4"/> Results</TabsTrigger>
              <TabsTrigger value="history" className="gap-2"><SlidersHorizontal className="h-4 w-4"/> Settings Snapshot</TabsTrigger>
            </TabsList>
            <TabsContent value="results" className="mt-4">
              {processedCount === 0 ? (
                <Card className="border-white/20 dark:border-white/10 bg-white/60 dark:bg-white/5">
                  <CardContent className="p-10 text-center text-neutral-500 dark:text-neutral-400">
                    Process some images to see your cutouts here.
                  </CardContent>
                </Card>
              ) : (
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
                  {items
                    .filter((i) => i.status === "done")
                    .map((i) => (
                      <Card key={`done-${i.id}`} className="overflow-hidden border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/5">
                        <div className="aspect-[4/3]">
                          <img src={i.cutoutUrl} alt="Cutout" className={`h-full w-full object-contain ${whiteBg ? "bg-white" : "bg-transparent"}`} />
                        </div>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-base truncate">{i.file.name}</CardTitle>
                          <CardDescription className="truncate">{model} • {enableUpscale ? `${i.upscale}×` : "no upscale"} • padding {padding}px</CardDescription>
                        </CardHeader>
                        <CardContent className="pb-4">
                          <div className="flex items-center gap-2">
                            <Button size="sm" className="gap-2"><Download className="h-4 w-4"/> Download</Button>
                            <Button size="sm" variant="secondary" className="gap-2"><Wand2 className="h-4 w-4"/> Fine‑Tune</Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                </div>
              )}
            </TabsContent>
            <TabsContent value="history" className="mt-4">
              <Card className="border-white/20 dark:border-white/10 bg-white/60 dark:bg-white/5">
                <CardContent className="p-6 text-sm">
                  <ul className="space-y-2">
                    <li className="flex items-center gap-2"><Check className="h-4 w-4"/> Model: {model}</li>
                    <li className="flex items-center gap-2"><Check className="h-4 w-4"/> Upscale: {enableUpscale ? `${scale}× (${upscaleMethod})` : "Disabled"}</li>
                    <li className="flex items-center gap-2"><Check className="h-4 w-4"/> Background: {whiteBg ? "White" : "Transparent"}</li>
                    <li className="flex items-center gap-2"><Check className="h-4 w-4"/> Padding: {padding}px</li>
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </section>

        {/* Footer CTA */}
        <footer className="border-t border-white/20 dark:border-white/10">
          <div className="mx-auto max-w-7xl px-4 py-10 grid md:grid-cols-2 gap-6 items-center">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">Ship your cutouts at scale</h2>
              <p className="text-neutral-600 dark:text-neutral-300">Batch processing • REST/GraphQL APIs • Team roles • Webhooks • S3/GCS export</p>
            </div>
            <div className="flex md:justify-end gap-3">
              <Button variant="secondary" className="gap-2"><Info className="h-4 w-4"/> Pricing</Button>
              <Button className="gap-2">Start Free <ChevronRight className="h-4 w-4"/></Button>
            </div>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}