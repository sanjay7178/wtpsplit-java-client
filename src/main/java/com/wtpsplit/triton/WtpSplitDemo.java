package com.wtpsplit.triton;

import java.util.List;

/**
 * WtpSplit Sentence Segmentation Demo.
 * 
 * Demonstrates sentence splitting using:
 * - Deep Java Library (DJL) for XLM-RoBERTa tokenization
 * - NVIDIA Triton gRPC for model inference
 * 
 * Usage:
 *   mvn exec:java -Dexec.args="localhost 8085 sat_3l_sm"
 */
public class WtpSplitDemo {
    
    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "localhost";
        int grpcPort = args.length > 1 ? Integer.parseInt(args[1]) : 8085;
        String modelName = args.length > 2 ? args[2] : "sat_3l_sm";
        
        printHeader(host, grpcPort, modelName);
        
        try (WtpSplit wtp = new WtpSplit(host, grpcPort, modelName)) {
            
            System.out.println("✓ Connected to Triton server\n");
            
            // Test sentences
            String[] testTexts = {
                "Hello world. This is a test. How are you doing today?",
                "The quick brown fox jumps over the lazy dog. It was a beautiful sunny day. The birds were singing.",
                "Dr. Smith went to Washington D.C. He met with Sen. Johnson at 3:30 p.m. They discussed the new policy.",
                "I love programming! Do you? It's really fun. Let's learn together.",
                "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
            };
            
            for (String text : testTexts) {
                printSentenceSplit(wtp, text);
            }
            
            // Interactive demo with longer text
            String longText = """
                Artificial intelligence is transforming the world. Machine learning models can now understand natural language. 
                This opens up many possibilities. Companies are using AI for customer service. They're also using it for data analysis. 
                The future looks promising. However, we must be careful about ethical considerations. 
                AI should be used responsibly. What do you think about this?
                """;
            
            System.out.println("─".repeat(60));
            System.out.println("LONGER TEXT EXAMPLE:");
            System.out.println("─".repeat(60));
            printSentenceSplit(wtp, longText.replace("\n", " ").trim());
            
        } catch (Exception e) {
            System.err.println("✗ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printHeader(String host, int port, String model) {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║         WtpSplit Sentence Segmentation Demo              ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.println("║  Server:    " + padRight(host + ":" + port, 44) + "║");
        System.out.println("║  Model:     " + padRight(model, 44) + "║");
        System.out.println("║  Tokenizer: " + padRight("XLM-RoBERTa (DJL)", 44) + "║");
        System.out.println("║  Protocol:  " + padRight("gRPC", 44) + "║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();
    }
    
    private static void printSentenceSplit(WtpSplit wtp, String text) {
        System.out.println("─".repeat(60));
        System.out.println("INPUT:");
        System.out.println("  \"" + truncate(text, 70) + "\"");
        System.out.println();
        
        long start = System.currentTimeMillis();
        List<String> sentences = wtp.split(text);
        long elapsed = System.currentTimeMillis() - start;
        
        System.out.println("OUTPUT (" + sentences.size() + " sentences, " + elapsed + "ms):");
        for (int i = 0; i < sentences.size(); i++) {
            String sentence = sentences.get(i).trim();
            System.out.println("  [" + (i + 1) + "] \"" + sentence + "\"");
        }
        System.out.println();
    }
    
    private static String padRight(String s, int n) {
        return String.format("%-" + n + "s", s);
    }
    
    private static String truncate(String s, int maxLen) {
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }
}

