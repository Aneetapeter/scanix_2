import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../widgets/navigation_bar.dart';
import '../widgets/footer_section.dart';

class HowItWorksScreen extends StatelessWidget {
  const HowItWorksScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Column(
          children: [
            const CustomNavigationBar(),
            _buildContent(context),
            const FooterSection(),
          ],
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 40),
          
          // Header
          const Text(
            'How It Works',
            style: TextStyle(
              fontSize: 36,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Our AI-powered detection process in 4 simple steps',
            style: TextStyle(
              fontSize: 20,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 40),

          // Steps
          _buildStep(
            stepNumber: 1,
            title: 'Upload Image',
            description: 'Upload a clear photo of the face or use your camera to capture an image in real-time.',
            icon: Icons.upload_rounded,
            color: const Color(0xFF3B82F6),
          ),
          
          const SizedBox(height: 32),
          
          _buildStep(
            stepNumber: 2,
            title: 'AI Preprocessing',
            description: 'Our system uses OpenCV to detect and align facial features, ensuring optimal analysis conditions.',
            icon: Icons.tune,
            color: const Color(0xFF8B5CF6),
          ),
          
          const SizedBox(height: 32),
          
          _buildStep(
            stepNumber: 3,
            title: 'CNN Analysis',
            description: 'Advanced Convolutional Neural Network analyzes facial symmetry and muscle patterns to detect paralysis.',
            icon: Icons.psychology,
            color: const Color(0xFF10B981),
          ),
          
          const SizedBox(height: 32),
          
          _buildStep(
            stepNumber: 4,
            title: 'Results & Recommendations',
            description: 'Get instant results with confidence scores and professional recommendations for next steps.',
            icon: Icons.assessment,
            color: const Color(0xFFF59E0B),
          ),
          
          const SizedBox(height: 60),

          // Technical Details
          Container(
            padding: const EdgeInsets.all(32),
            decoration: BoxDecoration(
              color: const Color(0xFFF8FAFC),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFE2E8F0)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Technical Details',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1F2937),
                  ),
                ),
                const SizedBox(height: 20),
                
                _buildTechDetail(
                  'AI Model',
                  'Convolutional Neural Network (CNN) trained on thousands of medical images',
                  Icons.smart_toy,
                ),
                
                const SizedBox(height: 16),
                
                _buildTechDetail(
                  'Image Processing',
                  'OpenCV for facial detection, alignment, and preprocessing',
                  Icons.image,
                ),
                
                const SizedBox(height: 16),
                
                _buildTechDetail(
                  'Accuracy',
                  '95%+ accuracy in detecting facial paralysis with confidence scoring',
                  Icons.analytics,
                ),
                
                const SizedBox(height: 16),
                
                _buildTechDetail(
                  'Processing Time',
                  'Results delivered in under 10 seconds for real-time analysis',
                  Icons.speed,
                ),
              ],
            ),
          ),
          
          const SizedBox(height: 40),

          // CTA Section
          Container(
            padding: const EdgeInsets.all(32),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF1E3A8A), Color(0xFF3B82F6)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              children: [
                const Text(
                  'Ready to Try It?',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Experience the power of AI in medical diagnosis',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white70,
                  ),
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: () => context.go('/detection'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: const Color(0xFF1E3A8A),
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text(
                    'Start Detection Now',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStep({
    required int stepNumber,
    required String title,
    required String description,
    required IconData icon,
    required Color color,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(30),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: Colors.white, size: 24),
              Text(
                stepNumber.toString(),
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 24),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1F2937),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                description,
                style: const TextStyle(
                  fontSize: 16,
                  color: Color(0xFF6B7280),
                  height: 1.6,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildTechDetail(String title, String description, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: const Color(0xFF1E3A8A), size: 20),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF1F2937),
                ),
              ),
              Text(
                description,
                style: const TextStyle(
                  fontSize: 14,
                  color: Color(0xFF6B7280),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
