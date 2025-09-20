import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import 'screens/modern_home_screen.dart';
import 'screens/enhanced_detection_screen.dart';
import 'screens/about_screen.dart';
import 'screens/doctors_screen.dart';
import 'screens/how_it_works_screen.dart';
import 'screens/contact_screen.dart';
import 'screens/faq_screen.dart';
import 'screens/login_screen.dart';
import 'screens/signup_screen.dart';
import 'screens/camera_screen.dart';
import 'services/api_service.dart';
import 'services/image_service.dart';
import 'services/auth_service.dart';
import 'utils/app_theme.dart';
import 'widgets/app_wrapper.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ScanixApp());
}

class ScanixApp extends StatelessWidget {
  const ScanixApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ApiService()),
        ChangeNotifierProvider(create: (_) => ImageService()),
        ChangeNotifierProvider(create: (_) => AuthService()),
      ],
      child: AppWrapper(
        child: MaterialApp.router(
          title: 'Scanix - AI Facial Paralysis Detection',
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.system,
          routerConfig: _router,
          debugShowCheckedModeBanner: false,
        ),
      ),
    );
  }
}

final GoRouter _router = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      path: '/',
      name: 'home',
      builder: (context, state) => const ModernHomeScreen(),
    ),
    GoRoute(
      path: '/detection',
      name: 'detection',
      builder: (context, state) => const EnhancedDetectionScreen(),
    ),
    GoRoute(
      path: '/about',
      name: 'about',
      builder: (context, state) => const AboutScreen(),
    ),
    GoRoute(
      path: '/doctors',
      name: 'doctors',
      builder: (context, state) => const DoctorsScreen(),
    ),
    GoRoute(
      path: '/how-it-works',
      name: 'how-it-works',
      builder: (context, state) => const HowItWorksScreen(),
    ),
    GoRoute(
      path: '/contact',
      name: 'contact',
      builder: (context, state) => const ContactScreen(),
    ),
    GoRoute(
      path: '/faq',
      name: 'faq',
      builder: (context, state) => const FaqScreen(),
    ),
    GoRoute(
      path: '/login',
      name: 'login',
      builder: (context, state) => const LoginScreen(),
    ),
    GoRoute(
      path: '/signup',
      name: 'signup',
      builder: (context, state) => const SignupScreen(),
    ),
    GoRoute(
      path: '/camera',
      name: 'camera',
      builder: (context, state) => const CameraScreen(),
    ),
  ],
);
