class DetectionResult {
  final bool hasParalysis;
  final double confidence;
  final String recommendation;
  final String? heatmapPath;
  final DateTime timestamp;

  DetectionResult({
    required this.hasParalysis,
    required this.confidence,
    required this.recommendation,
    this.heatmapPath,
    required this.timestamp,
  });

  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      hasParalysis: json['has_paralysis'] ?? false,
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      recommendation: json['recommendation'] ?? '',
      heatmapPath: json['heatmap_path'],
      timestamp: DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'has_paralysis': hasParalysis,
      'confidence': confidence,
      'recommendation': recommendation,
      'heatmap_path': heatmapPath,
      'timestamp': timestamp.toIso8601String(),
    };
  }
}
