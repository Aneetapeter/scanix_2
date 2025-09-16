import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../widgets/navigation_bar.dart';
import '../widgets/footer_section.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

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
            'About Scanix',
            style: TextStyle(
              fontSize: 36,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'AI-Powered Facial Paralysis Detection for Early Diagnosis',
            style: TextStyle(
              fontSize: 20,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 40),

          // What is Facial Paralysis Section
          _buildSection(
            'What is Facial Paralysis?',
            'Facial paralysis occurs when the facial nerve (cranial nerve VII) is damaged, resulting in weakness or paralysis of the facial muscles. This can affect one or both sides of the face, causing difficulty with facial expressions, eye closure, and speech.',
            Icons.medical_services,
          ),
          
          const SizedBox(height: 32),

          // Why Early Detection Matters
          _buildSection(
            'Why Early Detection Matters',
            'Early detection of facial paralysis is crucial for effective treatment. The sooner the condition is identified, the better the chances of recovery. Our AI-powered tool helps healthcare providers and patients identify potential facial paralysis quickly and accurately.',
            Icons.schedule,
          ),
          
          const SizedBox(height: 32),

          // How Our AI Works
          _buildSection(
            'How Our AI Model Works',
            'Our advanced Convolutional Neural Network (CNN) analyzes facial images to detect subtle asymmetries and muscle weakness patterns. The model has been trained on thousands of medical images and can identify facial paralysis with high accuracy.',
            Icons.psychology,
          ),
          
          const SizedBox(height: 32),

          // Technology Stack
          _buildSection(
            'Technology Stack',
            'Built with cutting-edge technology including Flutter for the frontend, Python Flask for the backend, OpenCV for image processing, and TensorFlow for machine learning. Our system ensures fast, secure, and accurate analysis.',
            Icons.code,
          ),
          
          const SizedBox(height: 32),

          // Disclaimer
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFFFEF3C7),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFFF59E0B)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(
                  children: [
                    Icon(Icons.warning, color: Color(0xFFF59E0B)),
                    SizedBox(width: 8),
                    Text(
                      'Important Disclaimer',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF92400E),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                const Text(
                  'This tool is designed to assist healthcare professionals and should not be used as a replacement for professional medical diagnosis. Always consult with a qualified healthcare provider for proper medical evaluation and treatment.',
                  style: TextStyle(
                    fontSize: 14,
                    color: Color(0xFF92400E),
                  ),
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
                  'Ready to Get Started?',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Try our AI-powered facial paralysis detection tool',
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
                    'Start Detection',
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

  Widget _buildSection(String title, String content, IconData icon) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: const Color(0xFF1E3A8A).withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(
            icon,
            color: const Color(0xFF1E3A8A),
            size: 32,
          ),
        ),
        const SizedBox(width: 20),
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
              const SizedBox(height: 12),
              Text(
                content,
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
}
