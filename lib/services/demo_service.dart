import 'dart:math';
import '../models/detection_result.dart';

class DemoService {
  static DetectionResult generateDemoResult() {
    final random = Random();
    final hasParalysis = random.nextBool();
    final confidence = hasParalysis 
        ? 0.6 + random.nextDouble() * 0.3  // 60-90% if paralysis detected
        : 0.1 + random.nextDouble() * 0.3; // 10-40% if normal
    
    String recommendation;
    if (hasParalysis) {
      if (confidence > 0.8) {
        recommendation = "High confidence of facial paralysis detected. Please consult a neurologist immediately for urgent evaluation and treatment.";
      } else {
        recommendation = "Potential facial paralysis detected. We recommend scheduling an appointment with a neurologist for further evaluation.";
      }
    } else {
      if (confidence < 0.2) {
        recommendation = "No signs of facial paralysis detected. Continue regular health monitoring.";
      } else {
        recommendation = "Low probability of facial paralysis. If you have concerns, consider consulting a healthcare provider.";
      }
    }
    
    return DetectionResult(
      hasParalysis: hasParalysis,
      confidence: confidence,
      recommendation: recommendation,
      timestamp: DateTime.now(),
    );
  }
}
