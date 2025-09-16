import 'package:flutter/material.dart';

class FeaturesSection extends StatelessWidget {
  const FeaturesSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 80),
      color: const Color(0xFFF8FAFC),
      child: Column(
        children: [
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
            'Simple steps to get AI-powered diagnosis',
            style: TextStyle(
              fontSize: 20,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 60),
          
          Row(
            children: [
              _buildFeature(
                icon: Icons.upload_rounded,
                title: 'Upload Image',
                description: 'Upload a clear photo or use your camera to capture an image in real-time.',
                color: const Color(0xFF3B82F6),
              ),
              const SizedBox(width: 40),
              _buildFeature(
                icon: Icons.tune,
                title: 'AI Preprocessing',
                description: 'Our system uses OpenCV to detect and align facial features for optimal analysis.',
                color: const Color(0xFF8B5CF6),
              ),
              const SizedBox(width: 40),
              _buildFeature(
                icon: Icons.psychology,
                title: 'CNN Analysis',
                description: 'Advanced neural network analyzes facial symmetry and muscle patterns.',
                color: const Color(0xFF10B981),
              ),
              const SizedBox(width: 40),
              _buildFeature(
                icon: Icons.assessment,
                title: 'Get Results',
                description: 'Receive instant results with confidence scores and recommendations.',
                color: const Color(0xFFF59E0B),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildFeature({
    required IconData icon,
    required String title,
    required String description,
    required Color color,
  }) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 8),
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    color.withOpacity(0.1),
                    color.withOpacity(0.2),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(40),
                border: Border.all(
                  color: color.withOpacity(0.2),
                  width: 2,
                ),
              ),
              child: Icon(
                icon,
                color: color,
                size: 40,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              title,
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1F2937),
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              description,
              style: const TextStyle(
                fontSize: 14,
                color: Color(0xFF6B7280),
                height: 1.6,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
