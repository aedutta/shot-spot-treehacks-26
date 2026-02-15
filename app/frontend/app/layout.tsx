import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: " | Dataset Factory",
  description: "Modal + Bright Data Video Ingestion Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#09090b] text-zinc-50 font-mono antialiased overflow-hidden selection:bg-zinc-800">
        <main className="flex h-screen w-full flex-col">
          {children}
        </main>
      </body>
    </html>
  );
}
