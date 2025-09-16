import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import '../services/api_service.dart';
import '../services/image_service.dart';
import '../services/demo_service.dart';
import '../models/detection_result.dart';
import '../widgets/image_upload_widget.dart';
import '../widgets/result_display_widget.dart';

import '../widgets/professional_loading.dart';
import '../widgets/professional_card.dart';

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  DetectionResult? _result;
  bool _isAnalyzing = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ImageService>().initializeCamera();
    });
  }

  @override
  void dispose() {
    context.read<ImageService>().disposeCamera();
    super.dispose();
  }

  Future<void> _analyzeImage() async {
    final imageService = context.read<ImageService>();
    final apiService = context.read<ApiService>();

    if (imageService.selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select an image first')),
      );
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _result = null;
    });

    // Try real API first
    DetectionResult? result = await apiService.analyzeImage(imageService.selectedImage!);
    
    // If API fails, use demo mode
    if (result == null) {
      // Simulate processing delay
      await Future.delayed(const Duration(seconds: 2));
      result = DemoService.generateDemoResult();
      
      // Show demo mode warning
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Demo Mode: Using simulated AI analysis. Backend server not available.'),
          backgroundColor: Colors.orange,
          duration: Duration(seconds: 3),
        ),
      );
    }
    
    setState(() {
      _isAnalyzing = false;
      _result = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Facial Paralysis Detection'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ProfessionalSection(
              title: 'AI Facial Analysis',
              subtitle: 'Upload an image for professional AI-powered analysis',
              icon: Icons.psychology,
              iconColor: const Color(0xFF0F172A),
              child: const ImageUploadWidget(),
            ),
            
            const SizedBox(height: 24),
            
            // Analyze Button
            Consumer<ImageService>(
              builder: (context, imageService, child) {
                return ElevatedButton(
                  onPressed: imageService.selectedImage != null && !_isAnalyzing
                      ? _analyzeImage
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1E3A8A),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: _isAnalyzing
                      ? const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                              ),
                            ),
                            SizedBox(width: 12),
                            Text('Analyzing...'),
                          ],
                        )
                      : const Text(
                          'Analyze Image',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                );
              },
            ),
            
            const SizedBox(height: 24),
            
            // Professional Loading
            if (_isAnalyzing) 
              const Center(
                child: ProfessionalLoading(
                  message: 'Analyzing Image',
                  size: 60,
                ),
              ),
            
            const SizedBox(height: 24),
            
            // Results Section
            if (_result != null) ResultDisplayWidget(result: _result!),
            
            const SizedBox(height: 32),
            
            // Action Buttons
            if (_result != null) ...[
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () {
                        context.read<ImageService>().clearSelectedImage();
                        setState(() {
                          _result = null;
                        });
                      },
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Text('Analyze Another'),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () => context.go('/doctors'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF10B981),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Text('Consult Doctor'),
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
