package com.wtpsplit.triton;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * WtpSplit Sentence Segmentation using Triton + DJL.
 * 
 * Uses XLM-RoBERTa tokenizer from Deep Java Library (DJL)
 * and gRPC inference via NVIDIA Triton.
 * 
 * Example:
 * <pre>
 * try (WtpSplit wtp = new WtpSplit("localhost", 8085, "sat_3l_sm")) {
 *     List&lt;String&gt; sentences = wtp.split("Hello world. This is a test.");
 *     sentences.forEach(System.out::println);
 * }
 * </pre>
 */
public class WtpSplit implements Closeable {
    
    private final TritonClient client;
    private final HuggingFaceTokenizer tokenizer;
    private final float threshold;
    private final int blockSize;
    private final int stride;
    
    // XLM-RoBERTa special tokens
    private static final int CLS_TOKEN = 0;
    private static final int SEP_TOKEN = 2;
    
    /**
     * Create WtpSplit with default settings.
     */
    public WtpSplit(String host, int grpcPort, String modelName) throws IOException {
        this(host, grpcPort, modelName, 0.25f, 512, 64);
    }
    
    /**
     * Create WtpSplit with custom settings.
     * 
     * @param host Triton server host
     * @param grpcPort Triton gRPC port
     * @param modelName Model name (e.g., "sat_3l_sm")
     * @param threshold Split probability threshold (default: 0.25)
     * @param blockSize Maximum sequence length (default: 512)
     * @param stride Sliding window stride (default: 64)
     */
    public WtpSplit(String host, int grpcPort, String modelName, 
                    float threshold, int blockSize, int stride) throws IOException {
        this.client = new TritonClient(host, grpcPort, modelName);
        this.tokenizer = HuggingFaceTokenizer.newInstance("xlm-roberta-base");
        this.threshold = threshold;
        this.blockSize = blockSize;
        this.stride = stride;
        
        if (!client.isReady()) {
            throw new IOException("Triton server or model not ready");
        }
    }
    
    /**
     * Split text into sentences.
     * 
     * @param text Input text
     * @return List of sentences
     */
    public List<String> split(String text) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        
        float[] charProbs = predictCharProbabilities(text);
        return extractSentences(text, charProbs);
    }
    
    /**
     * Get split probability for each character.
     * 
     * @param text Input text
     * @return Probability array (length = text.length())
     */
    public float[] predictCharProbabilities(String text) {
        // Tokenize
        Encoding encoding = tokenizer.encode(text);
        long[] tokenIds = encoding.getIds();
        String[] tokens = encoding.getTokens();
        
        int numTokens = tokenIds.length;
        int effectiveBlockSize = Math.min(blockSize - 2, numTokens);
        
        // Compute character positions for each token
        int[][] tokenCharPositions = computeTokenPositions(text, tokens);
        
        // Process with sliding window
        float[] tokenProbs = new float[numTokens];
        float[] tokenCounts = new float[numTokens];
        
        for (int start = 0; start < numTokens; start += stride) {
            int end = Math.min(start + effectiveBlockSize, numTokens);
            if (end == numTokens && start > 0) {
                start = Math.max(0, end - effectiveBlockSize);
            }
            
            // Build input with CLS and SEP
            int chunkLen = end - start;
            int[] inputIds = new int[chunkLen + 2];
            int[] attentionMask = new int[chunkLen + 2];
            
            inputIds[0] = CLS_TOKEN;
            attentionMask[0] = 1;
            for (int i = 0; i < chunkLen; i++) {
                inputIds[i + 1] = (int) tokenIds[start + i];
                attentionMask[i + 1] = 1;
            }
            inputIds[chunkLen + 1] = SEP_TOKEN;
            attentionMask[chunkLen + 1] = 1;
            
            // Run inference
            float[][] logits = client.infer(inputIds, attentionMask);
            
            // Accumulate probabilities (skip CLS and SEP)
            for (int i = 0; i < chunkLen; i++) {
                float prob = sigmoid(logits[i + 1][0]);
                tokenProbs[start + i] += prob;
                tokenCounts[start + i] += 1.0f;
            }
            
            if (end >= numTokens) break;
        }
        
        // Average token probabilities
        for (int i = 0; i < numTokens; i++) {
            if (tokenCounts[i] > 0) {
                tokenProbs[i] /= tokenCounts[i];
            }
        }
        
        // Map token probabilities to character probabilities
        return mapToCharProbabilities(text, tokenProbs, tokenCharPositions);
    }
    
    /**
     * Compute character positions for each token by matching in the text.
     */
    private int[][] computeTokenPositions(String text, String[] tokens) {
        int[][] positions = new int[tokens.length][2]; // [start, end]
        int currentPos = 0;
        
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
            
            // Handle XLM-RoBERTa's special space prefix (▁)
            String searchToken = token.replace("▁", " ").replace("Ġ", " ");
            if (searchToken.startsWith(" ") && currentPos == 0) {
                searchToken = searchToken.substring(1);
            }
            
            // Find the token in the text starting from current position
            int foundPos = -1;
            String lowerText = text.toLowerCase();
            String lowerToken = searchToken.toLowerCase();
            
            // Try exact match first
            if (currentPos < text.length()) {
                int idx = lowerText.indexOf(lowerToken, currentPos);
                if (idx >= 0 && idx <= currentPos + 10) {
                    foundPos = idx;
                }
            }
            
            if (foundPos >= 0) {
                positions[i][0] = foundPos;
                positions[i][1] = foundPos + searchToken.length();
                currentPos = positions[i][1];
            } else {
                // Fallback: advance by 1
                positions[i][0] = currentPos;
                positions[i][1] = Math.min(currentPos + 1, text.length());
                currentPos = positions[i][1];
            }
        }
        
        return positions;
    }
    
    /**
     * Map token-level probabilities to character-level.
     * The probability is assigned to the last character of each token.
     */
    private float[] mapToCharProbabilities(String text, float[] tokenProbs, int[][] positions) {
        float[] charProbs = new float[text.length()];
        
        for (int i = 0; i < tokenProbs.length; i++) {
            int endChar = positions[i][1];
            if (endChar > 0 && endChar <= text.length()) {
                charProbs[endChar - 1] = Math.max(charProbs[endChar - 1], tokenProbs[i]);
            }
        }
        
        return charProbs;
    }
    
    /**
     * Extract sentences from text using character probabilities.
     */
    private List<String> extractSentences(String text, float[] charProbs) {
        List<String> sentences = new ArrayList<>();
        int lastSplit = 0;
        
        for (int i = 0; i < text.length(); i++) {
            if (charProbs[i] > threshold) {
                String sentence = text.substring(lastSplit, i + 1);
                if (!sentence.trim().isEmpty()) {
                    sentences.add(sentence);
                }
                lastSplit = i + 1;
            }
        }
        
        // Add remaining text
        if (lastSplit < text.length()) {
            String remaining = text.substring(lastSplit);
            if (!remaining.trim().isEmpty()) {
                sentences.add(remaining);
            }
        }
        
        return sentences;
    }
    
    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
    
    public boolean isReady() {
        return client.isReady();
    }
    
    @Override
    public void close() throws IOException {
        client.close();
    }
}
