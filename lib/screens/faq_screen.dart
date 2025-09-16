import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../widgets/navigation_bar.dart';
import '../widgets/footer_section.dart';

class FaqScreen extends StatefulWidget {
  const FaqScreen({super.key});

  @override
  State<FaqScreen> createState() => _FaqScreenState();
}

class _FaqScreenState extends State<FaqScreen> {
  final List<FAQItem> _faqs = [
    FAQItem(
      question: 'What is facial paralysis?',
      answer:
          'Facial paralysis is a condition where the facial nerve (cranial nerve VII) is damaged, resulting in weakness or paralysis of the facial muscles. This can affect one or both sides of the face, causing difficulty with facial expressions, eye closure, and speech.',
    ),
    FAQItem(
      question: 'How accurate is the AI detection?',
      answer:
          'Our AI model has been trained on thousands of medical images and achieves over 95% accuracy in detecting facial paralysis. However, this tool is designed to assist healthcare professionals and should not be used as a replacement for professional medical diagnosis.',
    ),
    FAQItem(
      question: 'Can this replace a doctor?',
      answer:
          'No, this tool is designed to assist healthcare professionals and should not be used as a replacement for professional medical diagnosis. Always consult with a qualified healthcare provider for proper medical evaluation and treatment.',
    ),
    FAQItem(
      question: 'Is my data secure?',
      answer:
          'Yes, we take data security seriously. All images are processed securely and are not stored permanently. We use industry-standard encryption and follow HIPAA guidelines for medical data protection.',
    ),
    FAQItem(
      question: 'What types of images work best?',
      answer:
          'For best results, use clear, well-lit photos with the face centered and looking directly at the camera. Avoid photos with heavy shadows, extreme angles, or obstructions covering the face.',
    ),
    FAQItem(
      question: 'How long does the analysis take?',
      answer:
          'The AI analysis typically takes 5-10 seconds to complete. The exact time may vary depending on image size and server load.',
    ),
    FAQItem(
      question: 'Can I use this for children?',
      answer:
          'Yes, the AI model can analyze facial images of people of all ages. However, for pediatric cases, we recommend consulting with a pediatric neurologist for proper evaluation.',
    ),
    FAQItem(
      question: 'What should I do if paralysis is detected?',
      answer:
          'If the AI detects potential facial paralysis, we recommend consulting with a neurologist immediately. Early detection and treatment are crucial for the best outcomes.',
    ),
    FAQItem(
      question: 'Is this tool free to use?',
      answer:
          'Yes, our basic detection tool is free to use. We also offer premium features and telemedicine consultations for a fee.',
    ),
    FAQItem(
      question: 'How can I contact support?',
      answer:
          'You can contact our support team through the contact form on our website, email us at support@scanix.ai, or call us at +1 (555) 123-4567.',
    ),
  ];

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
            'Frequently Asked Questions',
            style: TextStyle(
              fontSize: 36,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Find answers to common questions about our AI-powered facial paralysis detection',
            style: TextStyle(fontSize: 20, color: Color(0xFF6B7280)),
          ),
          const SizedBox(height: 40),

          // FAQ List
          ..._faqs.asMap().entries.map((entry) {
            final index = entry.key;
            final faq = entry.value;
            return Padding(
              padding: EdgeInsets.only(
                bottom: index < _faqs.length - 1 ? 16 : 0,
              ),
              child: _buildFAQItem(faq),
            );
          }),

          const SizedBox(height: 40),

          // Disclaimer
          Container(
            padding: const EdgeInsets.all(24),
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
                      'Medical Disclaimer',
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
                  'This tool is designed to assist healthcare professionals and should not be used as a replacement for professional medical diagnosis. The results provided are for informational purposes only and should not be considered as medical advice. Always consult with a qualified healthcare provider for proper medical evaluation and treatment.',
                  style: TextStyle(
                    fontSize: 14,
                    color: Color(0xFF92400E),
                    height: 1.6,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 40),

          // Privacy Policy
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFFF0F9FF),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFF0EA5E9)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(
                  children: [
                    Icon(Icons.privacy_tip, color: Color(0xFF0EA5E9)),
                    SizedBox(width: 8),
                    Text(
                      'Privacy Policy',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF0C4A6E),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                const Text(
                  'We are committed to protecting your privacy and personal information. All images uploaded for analysis are processed securely and are not stored permanently. We follow HIPAA guidelines for medical data protection and use industry-standard encryption to secure your data.',
                  style: TextStyle(
                    fontSize: 14,
                    color: Color(0xFF0C4A6E),
                    height: 1.6,
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
                  'Still Have Questions?',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Contact our support team for personalized assistance',
                  style: TextStyle(fontSize: 16, color: Colors.white70),
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: () => context.go('/contact'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: const Color(0xFF1E3A8A),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 32,
                      vertical: 16,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text(
                    'Contact Support',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFAQItem(FAQItem faq) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ExpansionTile(
        title: Text(
          faq.question,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Color(0xFF1F2937),
          ),
        ),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Text(
              faq.answer,
              style: const TextStyle(
                fontSize: 14,
                color: Color(0xFF6B7280),
                height: 1.6,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class FAQItem {
  final String question;
  final String answer;

  FAQItem({required this.question, required this.answer});
}
