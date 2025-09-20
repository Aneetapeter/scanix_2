import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class FooterSection extends StatelessWidget {
  const FooterSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 60),
      decoration: const BoxDecoration(
        color: Color(0xFF1F2937),
      ),
      child: Column(
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Logo and Description
              Expanded(
                flex: 2,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(
                          Icons.medical_services,
                          color: Colors.white,
                          size: 32,
                        ),
                        SizedBox(width: 8),
                        Text(
                          'Scanix',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'AI-powered facial paralysis detection for early diagnosis and better patient outcomes.',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.white70,
                        height: 1.6,
                      ),
                    ),
                    const SizedBox(height: 24),
                    Row(
                      children: [
                        _buildSocialIcon(Icons.facebook, () {}),
                        const SizedBox(width: 12),
                        _buildSocialIcon(Icons.work, () {}),
                        const SizedBox(width: 12),
                        _buildSocialIcon(Icons.chat, () {}),
                        const SizedBox(width: 12),
                        _buildSocialIcon(Icons.email, () {}),
                      ],
                    ),
                  ],
                ),
              ),
              
              const SizedBox(width: 60),
              
              // Quick Links
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Quick Links',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildFooterLink('Home', () => context.go('/')),
                    _buildFooterLink('Detection Tool', () => context.go('/detection')),
                    _buildFooterLink('About Us', () => context.go('/about')),
                    _buildFooterLink('Doctors', () => context.go('/doctors')),
                    _buildFooterLink('How It Works', () => context.go('/how-it-works')),
                    _buildFooterLink('Contact', () => context.go('/contact')),
                    _buildFooterLink('FAQ', () => context.go('/faq')),
                  ],
                ),
              ),
              
              const SizedBox(width: 60),
              
              // Contact Info
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Contact Info',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildContactInfo(Icons.email, 'support@scanix.ai'),
                    _buildContactInfo(Icons.phone, '+1 (555) 123-4567'),
                    _buildContactInfo(Icons.location_on, '123 Medical Plaza\nSan Francisco, CA 94102'),
                    _buildContactInfo(Icons.schedule, 'Mon-Fri: 9AM-6PM\nSat: 10AM-4PM'),
                  ],
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 40),
          
          const Divider(color: Colors.white24),
          
          const SizedBox(height: 24),
          
          // Bottom Section
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '© 2024 Scanix. All rights reserved.',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.white70,
                ),
              ),
              Row(
                children: [
                  TextButton(
                    onPressed: () {},
                    child: const Text(
                      'Privacy Policy',
                      style: TextStyle(color: Colors.white70),
                    ),
                  ),
                  const SizedBox(width: 20),
                  TextButton(
                    onPressed: () {},
                    child: const Text(
                      'Terms of Service',
                      style: TextStyle(color: Colors.white70),
                    ),
                  ),
                  const SizedBox(width: 20),
                  TextButton(
                    onPressed: () {},
                    child: const Text(
                      'Disclaimer',
                      style: TextStyle(color: Colors.white70),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSocialIcon(IconData icon, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _buildFooterLink(String text, VoidCallback onTap) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        onTap: onTap,
        child: Text(
          text,
          style: const TextStyle(
            fontSize: 14,
            color: Colors.white70,
          ),
        ),
      ),
    );
  }

  Widget _buildContactInfo(IconData icon, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: Colors.white70, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                fontSize: 14,
                color: Colors.white70,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
