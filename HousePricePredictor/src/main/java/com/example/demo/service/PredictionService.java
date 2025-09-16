package com.example.demo.service;

import ai.onnxruntime.*;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Collections;
import java.util.Map;

@Service
public class PredictionService {

    private OrtEnvironment env;
    private OrtSession session;

    // Mappings must be identical to the Python LabelEncoder
    private final Map<String, Float> locationMapping = Map.of("Downtown", 0f, "Rural", 1f, "Suburban", 2f, "Urban", 3f);
    private final Map<String, Float> conditionMapping = Map.of("Excellent", 0f, "Fair", 1f, "Good", 2f, "Poor", 3f);
    private final Map<String, Float> garageMapping = Map.of("No", 0f, "Yes", 1f);

    @PostConstruct // This method runs when the service is created
    public void init() throws OrtException {
        env = OrtEnvironment.getEnvironment();
        try {
            // Load the model from the resources folder
            InputStream modelStream = getClass().getClassLoader().getResourceAsStream("house_price_model.onnx");
            if (modelStream == null) {
                throw new RuntimeException("Model file not found in resources!");
            }
            // ONNX Runtime needs a file path, so we copy the stream to a temporary file
            Path tempFile = Files.createTempFile("onnx_model", ".tmp");
            Files.copy(modelStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            
            session = env.createSession(tempFile.toString(), new OrtSession.SessionOptions());
            
            // Clean up the temporary file
            Files.delete(tempFile);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    // In PredictionService.java

    public double predict(Map<String, Object> inputData) throws OrtException {
    // 1. Preprocess the input data
    float[] features = new float[8];

    // --- CORRECTED CODE BLOCK ---
    // Use Float.parseFloat() to convert the String from the form into a float
    features[0] = Float.parseFloat((String) inputData.get("Area"));
    features[1] = Float.parseFloat((String) inputData.get("Bedrooms"));
    features[2] = Float.parseFloat((String) inputData.get("Bathrooms"));
    features[3] = Float.parseFloat((String) inputData.get("Floors"));
    features[4] = Float.parseFloat((String) inputData.get("YearBuilt"));
    // --- END CORRECTED CODE BLOCK ---

    // The mapping for categorical features is correct as is
    features[5] = locationMapping.get((String) inputData.get("Location"));
    features[6] = conditionMapping.get((String) inputData.get("Condition"));
    features[7] = garageMapping.get((String) inputData.get("Garage"));

    // 2. Create the input tensor for the model
    // The shape is [1, 8] for 1 prediction with 8 features
    OnnxTensor inputTensor = OnnxTensor.createTensor(env, new float[][]{features});

    // 3. Run the prediction
    // Make sure "float_input" is the correct input name for your model
    OrtSession.Result result = session.run(Collections.singletonMap("float_input", inputTensor));

    // 4. Extract the result
    float[][] prediction = (float[][]) result.get(0).getValue();

    // Close tensor to release resources
    inputTensor.close();

    return prediction[0][0];
}
    @PreDestroy // This method runs when the application is shutting down
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
        if (env != null) {
            env.close();
        }
    }
}