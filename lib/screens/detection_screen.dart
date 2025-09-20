import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import '../services/api_service.dart';
import '../services/image_service.dart';
import '../models/detection_result.dart';
import '../widgets/image_upload_widget.dart';
import '../widgets/result_display_widget.dart';
import '../utils/app_theme.dart';
import '../widgets/professional_card.dart';
import '../widgets/professional_loading.dart';

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  DetectionResult? _result;
  bool _isAnalyzing = false;
  // Animation controllers removed as they're not used in this simplified version

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

    if (kDebugMode) {
      debugPrint('🔍 Starting image analysis...');
      debugPrint('Selected image: ${imageService.selectedImage}');
      debugPrint(
        'Selected image bytes: ${imageService.selectedImageBytes != null ? "Available" : "null"}',
      );
    }

    if (imageService.selectedImage == null &&
        imageService.selectedImageBytes == null) {
      if (kDebugMode) {
        debugPrint('❌ No image selected');
      }
      _showErrorSnackBar('Please select an image first');
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _result = null;
    });

    try {
      // Try real API first - pass the appropriate image data
      dynamic imageData =
          imageService.selectedImageBytes ?? imageService.selectedImage;
      if (kDebugMode) {
        debugPrint('📤 Sending image data to API: ${imageData.runtimeType}');
      }
      DetectionResult? result = await apiService.analyzeImage(imageData);
      if (kDebugMode) {
        debugPrint(
          '📥 Received result: ${result != null ? "Success" : "null"}',
        );
      }

      setState(() {
        _isAnalyzing = false;
        _result = result;
      });
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
      });

      if (mounted) {
        _showErrorSnackBar('Analysis failed: $e');
      }
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.white),
            const SizedBox(width: 8),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: AppTheme.errorRed,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
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
                  onPressed:
                      (kIsWeb
                              ? imageService.selectedImageBytes != null
                              : imageService.selectedImage != null) &&
                          !_isAnalyzing
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
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  Colors.white,
                                ),
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
