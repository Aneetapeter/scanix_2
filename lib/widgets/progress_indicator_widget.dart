import 'package:flutter/material.dart';

class ProgressIndicatorWidget extends StatelessWidget {
  const ProgressIndicatorWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const Text(
              'Analyzing Image',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1F2937),
              ),
            ),
            const SizedBox(height: 24),
            
            // Progress Steps
            Row(
              children: [
                _buildStep('Uploading', true, Icons.upload),
                _buildConnector(),
                _buildStep('Preprocessing', true, Icons.tune),
                _buildConnector(),
                _buildStep('Analyzing', true, Icons.psychology),
                _buildConnector(),
                _buildStep('Results', false, Icons.assessment),
              ],
            ),
            
            const SizedBox(height: 32),
            
            // Progress Bar
            const LinearProgressIndicator(
              backgroundColor: Color(0xFFE5E7EB),
              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF1E3A8A)),
            ),
            
            const SizedBox(height: 16),
            
            const Text(
              'This may take a few seconds...',
              style: TextStyle(
                fontSize: 14,
                color: Color(0xFF6B7280),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStep(String label, bool isActive, IconData icon) {
    return Expanded(
      child: Column(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: isActive ? const Color(0xFF1E3A8A) : const Color(0xFFE5E7EB),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Icon(
              icon,
              color: isActive ? Colors.white : const Color(0xFF9CA3AF),
              size: 20,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
              color: isActive ? const Color(0xFF1E3A8A) : const Color(0xFF9CA3AF),
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildConnector() {
    return Container(
      height: 2,
      width: 20,
      color: const Color(0xFFE5E7EB),
    );
  }
}
