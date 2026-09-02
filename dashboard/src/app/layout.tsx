import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SPEC-TCNN · Delay-to-lick cortico-striatal selection",
  description:
    "Selective Predictive Epoch Context: dilated-causal TCNN with attention and wavelet/STFT features for ALM–striatum delay codes.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#f6f3ee] text-stone-900">
        {children}
      </body>
    </html>
  );
}
