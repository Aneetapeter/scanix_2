import 'package:flutter/material.dart';
import '../widgets/navigation_bar.dart';
import '../widgets/hero_section.dart';
import '../widgets/features_section.dart';
import '../widgets/testimonials_section.dart';
import '../widgets/footer_section.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Column(
          children: [
            const CustomNavigationBar(),
            const HeroSection(),
            const FeaturesSection(),
            const TestimonialsSection(),
            const FooterSection(),
          ],
        ),
      ),
    );
  }
}
