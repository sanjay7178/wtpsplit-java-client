package com.wtpsplit.triton;

import com.google.protobuf.ByteString;
import inference.GRPCInferenceServiceGrpc;
import inference.GrpcService.ModelInferRequest;
import inference.GrpcService.ModelInferResponse;
import inference.GrpcService.ModelReadyRequest;
import inference.GrpcService.ServerReadyRequest;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;

/**
 * gRPC client for NVIDIA Triton Inference Server.
 * Optimized for wtpsplit sentence segmentation models.
 */
public class TritonClient implements Closeable {
    
    private final ManagedChannel channel;
    private final GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub;
    private final String modelName;
    
    public TritonClient(String host, int grpcPort, String modelName) {
        this.modelName = modelName;
        this.channel = ManagedChannelBuilder.forAddress(host, grpcPort)
                .usePlaintext()
                .maxInboundMessageSize(64 * 1024 * 1024)
                .build();
        this.stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
    }
    
    public boolean isReady() {
        try {
            boolean serverReady = stub.serverReady(ServerReadyRequest.newBuilder().build()).getReady();
            boolean modelReady = stub.modelReady(ModelReadyRequest.newBuilder().setName(modelName).build()).getReady();
            return serverReady && modelReady;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Run inference and return logits.
     * 
     * @param inputIds XLM-RoBERTa token IDs
     * @param attentionMask Attention mask (1 for valid tokens)
     * @return Logits array [seq_len, 1]
     */
    public float[][] infer(int[] inputIds, int[] attentionMask) {
        int seqLen = inputIds.length;
        
        // Convert to bytes
        byte[] inputBytes = toBytes(inputIds);
        byte[] maskBytes = toBytes(attentionMask);
        
        // Build request
        ModelInferRequest request = ModelInferRequest.newBuilder()
            .setModelName(modelName)
            .addInputs(ModelInferRequest.InferInputTensor.newBuilder()
                .setName("input_ids")
                .setDatatype("INT64")
                .addShape(1)
                .addShape(seqLen))
            .addInputs(ModelInferRequest.InferInputTensor.newBuilder()
                .setName("attention_mask")
                .setDatatype("INT64")
                .addShape(1)
                .addShape(seqLen))
            .addOutputs(ModelInferRequest.InferRequestedOutputTensor.newBuilder()
                .setName("logits"))
            .addRawInputContents(ByteString.copyFrom(inputBytes))
            .addRawInputContents(ByteString.copyFrom(maskBytes))
            .build();
        
        // Execute
        ModelInferResponse response = stub.modelInfer(request);
        
        // Parse response
        var output = response.getOutputs(0);
        int dim1 = (int) output.getShape(1);
        int dim2 = output.getShapeCount() > 2 ? (int) output.getShape(2) : 1;
        
        byte[] data = response.getRawOutputContents(0).toByteArray();
        float[] flat = "FP16".equals(output.getDatatype()) 
            ? fromFp16(data, dim1 * dim2) 
            : fromFp32(data, dim1 * dim2);
        
        // Reshape to [seq_len, 1]
        float[][] logits = new float[dim1][dim2];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                logits[i][j] = flat[i * dim2 + j];
            }
        }
        return logits;
    }
    
    private byte[] toBytes(int[] arr) {
        // Previously this serialized 4-byte ints (INT32) which caused a mismatch when the model
        // expects INT64. Serialize as 8-byte little-endian longs to match "INT64" datatype.
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : arr) buf.putLong((long) v);
        return buf.array();
    }
    
    private float[] fromFp32(byte[] bytes, int count) {
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[count];
        for (int i = 0; i < count; i++) out[i] = buf.getFloat();
        return out;
    }
    
    private float[] fromFp16(byte[] bytes, int count) {
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            int h = buf.getShort() & 0xFFFF;
            int sign = (h >> 15) & 1;
            int exp = (h >> 10) & 0x1F;
            int mant = h & 0x3FF;
            float val;
            if (exp == 0) val = mant == 0 ? 0f : (float)(mant / 1024.0 * Math.pow(2, -14));
            else if (exp == 31) val = mant == 0 ? Float.POSITIVE_INFINITY : Float.NaN;
            else val = (float)((1.0 + mant / 1024.0) * Math.pow(2, exp - 15));
            out[i] = sign == 1 ? -val : val;
        }
        return out;
    }
    
    @Override
    public void close() throws IOException {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}

