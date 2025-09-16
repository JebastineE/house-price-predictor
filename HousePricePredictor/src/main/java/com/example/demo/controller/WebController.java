package com.example.demo.controller; // Corrected package name

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import com.example.demo.service.PredictionService;

import java.util.Map;

@Controller
public class WebController {

    @Autowired
    private PredictionService predictionService;

    @GetMapping("/")
    public String showForm() {
        return "index"; // Renders index.html
    }

    @PostMapping("/predict")
    public String predict(@RequestParam Map<String, Object> formData, Model model) {
        try {
            double price = predictionService.predict(formData);
            String formattedPrice = String.format("₹%,.2f", price);
            model.addAttribute("predictionText", formattedPrice);
        } catch (Exception e) {
            // It's good practice to log the error
            e.printStackTrace();
            model.addAttribute("predictionText", "Error during prediction.");
        }
        return "result"; // Renders result.html
    }
}