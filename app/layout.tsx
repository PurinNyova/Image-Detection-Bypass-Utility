import type { Metadata } from "next";
import { DM_Sans, Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-heading",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "Image Detection Bypass Utility",
  description:
    "Upload an image, build an ordered pipeline of processing steps, run them sequentially, and download the final output.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html:
              'try{if(localStorage.theme==="dark"||(!localStorage.theme&&matchMedia("(prefers-color-scheme: dark)").matches))document.documentElement.classList.add("dark")}catch(e){}',
          }}
        />
      </head>
      <body className={`${spaceGrotesk.variable} ${dmSans.variable} bg-bg text-text antialiased`}>
        {children}
      </body>
    </html>
  );
}
