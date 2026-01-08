package com.wtpsplit.triton;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * Concurrent Benchmark for WtpSplit Triton gRPC Client.
 * 
 * Simulates production load by running multiple concurrent clients
 * making inference requests to the Triton server.
 * 
 * Usage:
 * mvn exec:java -Dexec.mainClass="com.wtpsplit.triton.ConcurrentBenchmark" \
 * -Dexec.args="localhost 8085 sat_3l_sm 20"
 */
public class ConcurrentBenchmark {

    // Sample text for benchmarking (medium length, ~350 chars)
    private static final String SAMPLE_TEXT = """
            Machine learning is a subset of artificial intelligence. It enables computers to learn from data.
            Deep learning is a subset of machine learning. Neural networks are the foundation of deep learning.
            These technologies have revolutionized many industries. Natural language processing is one application area.
            """;

    // Benchmark configuration
    private static final int[] CLIENT_COUNTS = { 1, 2, 4, 8, 16, 32 };
    private static final int WARMUP_ITERATIONS = 10;

    private final String host;
    private final int port;
    private final String modelName;

    private final int requestsPerClient;

    public ConcurrentBenchmark(String host, int port, String modelName, int requestsPerClient) {
        this.host = host;
        this.port = port;
        this.modelName = modelName;
        this.requestsPerClient = requestsPerClient;
    }

    /**
     * Run benchmark with multiple concurrent clients.
     */
    public void run() {
        int textLength = SAMPLE_TEXT.length();

        printHeader();
        System.out.printf("  Requests per client: %d%n", requestsPerClient);
        System.out.printf("  Text length: %d chars%n", textLength);
        System.out.println("=".repeat(80));

        // Warmup
        System.out.println("\nWarming up...");
        try (WtpSplit wtp = new WtpSplit(host, port, modelName)) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                wtp.split(SAMPLE_TEXT);
            }
            System.out.println("Warmup complete.\n");
        } catch (IOException e) {
            System.err.println("Failed to connect to Triton: " + e.getMessage());
            return;
        }

        // Print table header
        System.out.printf("  %8s | %12s | %15s | %12s | %12s%n",
                "Clients", "Total Time", "Throughput", "Avg Latency", "P99 Latency");
        System.out.println("  " + "-".repeat(70));

        // Run benchmark for each client count
        for (int numClients : CLIENT_COUNTS) {
            BenchmarkResult result = runWithClients(numClients);
            if (result != null) {
                System.out.printf("  %8d | %10.2f s | %12.0f c/s | %10.2f ms | %10.2f ms%n",
                        numClients,
                        result.totalTimeSec,
                        result.throughputCharsPerSec,
                        result.avgLatencyMs,
                        result.p99LatencyMs);
            } else {
                System.out.printf("  %8d | %12s | %15s | %12s | %12s%n",
                        numClients, "FAILED", "N/A", "N/A", "N/A");
            }
        }

        System.out.println("=".repeat(80));
        System.out.println("\nBenchmark complete!");
    }

    /**
     * Run benchmark with specified number of concurrent clients.
     */
    private BenchmarkResult runWithClients(int numClients) {
        ExecutorService executor = Executors.newFixedThreadPool(numClients);
        List<Future<List<Double>>> futures = new ArrayList<>();

        long startTime = System.nanoTime();

        // Submit tasks for each client
        for (int i = 0; i < numClients; i++) {
            futures.add(executor.submit(this::clientWorker));
        }

        // Collect all latencies
        List<Double> allLatencies = new ArrayList<>();
        boolean failed = false;

        for (Future<List<Double>> future : futures) {
            try {
                allLatencies.addAll(future.get());
            } catch (Exception e) {
                System.err.println("Client failed: " + e.getMessage());
                failed = true;
            }
        }

        long endTime = System.nanoTime();
        executor.shutdown();

        if (failed || allLatencies.isEmpty()) {
            return null;
        }

        // Calculate metrics
        double totalTimeSec = (endTime - startTime) / 1_000_000_000.0;
        int totalRequests = numClients * requestsPerClient;
        long totalChars = (long) totalRequests * SAMPLE_TEXT.length();
        double throughput = totalChars / totalTimeSec;
        double avgLatency = allLatencies.stream().mapToDouble(d -> d).average().orElse(0);
        double p99Latency = percentile(allLatencies, 99);

        return new BenchmarkResult(totalTimeSec, throughput, avgLatency, p99Latency);
    }

    /**
     * Single client worker that makes multiple requests.
     */
    private List<Double> clientWorker() {
        List<Double> latencies = new ArrayList<>();

        try (WtpSplit wtp = new WtpSplit(host, port, modelName)) {
            for (int i = 0; i < requestsPerClient; i++) {
                long start = System.nanoTime();
                wtp.split(SAMPLE_TEXT);
                long end = System.nanoTime();

                double latencyMs = (end - start) / 1_000_000.0;
                latencies.add(latencyMs);
            }
        } catch (IOException e) {
            throw new RuntimeException("Client connection failed: " + e.getMessage(), e);
        }

        return latencies;
    }

    /**
     * Calculate percentile from a list of values.
     */
    private double percentile(List<Double> values, double percentile) {
        if (values.isEmpty())
            return 0;

        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);

        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        index = Math.max(0, Math.min(index, sorted.size() - 1));

        return sorted.get(index);
    }

    private void printHeader() {
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println(" WtpSplit Java gRPC Concurrent Benchmark");
        System.out.println("=".repeat(80));
        System.out.printf("  Host: %s:%d%n", host, port);
        System.out.printf("  Model: %s%n", modelName);
    }

    /**
     * Benchmark result container.
     */
    private static class BenchmarkResult {
        final double totalTimeSec;
        final double throughputCharsPerSec;
        final double avgLatencyMs;
        final double p99LatencyMs;

        BenchmarkResult(double totalTimeSec, double throughputCharsPerSec,
                double avgLatencyMs, double p99LatencyMs) {
            this.totalTimeSec = totalTimeSec;
            this.throughputCharsPerSec = throughputCharsPerSec;
            this.avgLatencyMs = avgLatencyMs;
            this.p99LatencyMs = p99LatencyMs;
        }
    }

    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 8085;
        String modelName = args.length > 2 ? args[2] : "sat_3l_sm";
        int requestsPerClient = args.length > 3 ? Integer.parseInt(args[3]) : 20;

        ConcurrentBenchmark benchmark = new ConcurrentBenchmark(host, port, modelName, requestsPerClient);
        benchmark.run();
    }
}