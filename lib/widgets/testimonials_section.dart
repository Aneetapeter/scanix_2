import 'package:flutter/material.dart';

class TestimonialsSection extends StatelessWidget {
  const TestimonialsSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 80),
      child: Column(
        children: [
          const Text(
            'What Healthcare Professionals Say',
            style: TextStyle(
              fontSize: 36,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Trusted by doctors and medical professionals worldwide',
            style: TextStyle(
              fontSize: 20,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 60),
          
          Row(
            children: [
              _buildTestimonial(
                name: 'Dr. Sarah Johnson',
                role: 'Neurologist, Mayo Clinic',
                content: 'This tool has revolutionized our early detection process. The AI accuracy is impressive and has helped us identify stroke symptoms quickly in rural clinics.',
                rating: 5,
              ),
              const SizedBox(width: 32),
              _buildTestimonial(
                name: 'Dr. Michael Chen',
                role: 'Telemedicine Specialist',
                content: 'The integration with our telemedicine platform is seamless. Patients can get preliminary assessments before their video consultations.',
                rating: 5,
              ),
              const SizedBox(width: 32),
              _buildTestimonial(
                name: 'Dr. Emily Rodriguez',
                role: 'Emergency Medicine',
                content: 'In emergency situations, every second counts. This tool helps us make faster, more accurate decisions about facial paralysis cases.',
                rating: 5,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildTestimonial({
    required String name,
    required String role,
    required String content,
    required int rating,
  }) {
    return Expanded(
      child: Card(
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Stars
              Row(
                children: List.generate(5, (index) {
                  return Icon(
                    index < rating ? Icons.star : Icons.star_border,
                    color: const Color(0xFFF59E0B),
                    size: 20,
                  );
                }),
              ),
              const SizedBox(height: 16),
              
              // Content
              Text(
                '"$content"',
                style: const TextStyle(
                  fontSize: 16,
                  color: Color(0xFF6B7280),
                  height: 1.6,
                  fontStyle: FontStyle.italic,
                ),
              ),
              const SizedBox(height: 20),
              
              // Author
              Row(
                children: [
                  CircleAvatar(
                    backgroundColor: const Color(0xFF1E3A8A).withOpacity(0.1),
                    child: const Icon(
                      Icons.person,
                      color: Color(0xFF1E3A8A),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          name,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF1F2937),
                          ),
                        ),
                        Text(
                          role,
                          style: const TextStyle(
                            fontSize: 14,
                            color: Color(0xFF6B7280),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
