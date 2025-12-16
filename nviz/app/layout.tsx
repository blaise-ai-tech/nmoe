import type { Metadata } from "next";
import { B612, B612_Mono } from "next/font/google";
import "./globals.css";
import { RunProvider } from "@/lib/run-context";
import { ThemeProvider } from "@/components/theme-provider";

const b612Sans = B612({
  variable: "--font-b612-sans",
  subsets: ["latin"],
  weight: ["400", "700"],
});

const b612Mono = B612_Mono({
  variable: "--font-b612-mono",
  subsets: ["latin"],
  weight: ["400", "700"],
});

export const metadata: Metadata = {
  title: "nviz - Training Visualization",
  description: "SQLite-backed training viewer (minimal, fast, SSR)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${b612Sans.variable} ${b612Mono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          themes={["light", "rose-pine", "tomorrow-night-bright"]}
          enableSystem={false}
          storageKey="nviz-theme"
        >
          <RunProvider>
            {children}
          </RunProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
