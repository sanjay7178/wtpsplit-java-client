# WtpSplit Java Client

Java client for wtpsplit sentence segmentation using NVIDIA Triton gRPC and DJL tokenization.

## Features

- **XLM-RoBERTa tokenization** via Deep Java Library (DJL)
- **gRPC inference** via NVIDIA Triton Inference Server
- **FP16 support** for TensorRT models
- **Clean API** - just 3 Java files, ~500 lines total

## Quick Start

### Prerequisites

1. Triton server running with the wtpsplit model:
   ```bash
   tritonserver --model-repository=/home/ec2-user/wtpsplit/triton_models
   ```

2. Java 17+ and Maven

### Build & Run

```bash
cd /home/ec2-user/wtpsplit-java-client
mvn compile
mvn exec:java -Dexec.args="localhost 8085 sat_3l_sm"
```

### Output

```
╔══════════════════════════════════════════════════════════╗
║         WtpSplit Sentence Segmentation Demo              ║
╠══════════════════════════════════════════════════════════╣
║  Server:    localhost:8085                              ║
║  Model:     sat_3l_sm                                   ║
║  Tokenizer: XLM-RoBERTa (DJL)                           ║
║  Protocol:  gRPC                                        ║
╚══════════════════════════════════════════════════════════╝

✓ Connected to Triton server

INPUT:
  "Hello world. This is a test. How are you doing today?"

OUTPUT (3 sentences, 97ms):
  [1] "Hello world."
  [2] "This is a test."
  [3] "How are you doing today?"
```

## API Usage

```java
import com.wtpsplit.triton.WtpSplit;
import java.util.List;

try (WtpSplit wtp = new WtpSplit("localhost", 8085, "sat_3l_sm")) {
    List<String> sentences = wtp.split("Hello world. This is a test.");
    sentences.forEach(System.out::println);
}
// Output:
// Hello world.
// This is a test.
```

### Custom Settings

```java
WtpSplit wtp = new WtpSplit(
    "localhost",  // host
    8085,         // gRPC port
    "sat_3l_sm",  // model name
    0.25f,        // threshold (0.0-1.0)
    512,          // block size
    64            // stride
);
```

### Get Raw Probabilities

```java
float[] probs = wtp.predictCharProbabilities("Hello world.");
// probs[i] = probability of sentence boundary after character i
```

## Project Structure

```
wtpsplit-java-client/
├── pom.xml                    # Maven config with DJL + gRPC
├── src/main/
│   ├── java/com/wtpsplit/triton/
│   │   ├── TritonClient.java  # gRPC client (142 lines)
│   │   ├── WtpSplit.java      # Main API (249 lines)
│   │   └── WtpSplitDemo.java  # Demo (100 lines)
│   └── proto/
│       ├── grpc_service.proto
│       ├── model_config.proto
│       └── health.proto
```

## Performance

| Metric | Value |
|--------|-------|
| First inference (cold) | ~100ms |
| Subsequent inference | ~5-20ms |
| Throughput | ~50-100 sentences/sec |

## Dependencies

- **DJL HuggingFace Tokenizers** - XLM-RoBERTa tokenization
- **gRPC Java** - Triton communication
- **Protocol Buffers** - Triton proto definitions

## Comparison with Python

| Python | Java |
|--------|------|
| `from wtpsplit import SaT` | `new WtpSplit(...)` |
| `sat.split(text)` | `wtp.split(text)` |
| `tritonclient.grpc` | `TritonClient` (gRPC) |
| HuggingFace tokenizers | DJL HuggingFace tokenizers |
