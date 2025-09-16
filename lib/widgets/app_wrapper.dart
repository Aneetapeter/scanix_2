import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';

class AppWrapper extends StatefulWidget {
  final Widget child;

  const AppWrapper({
    super.key,
    required this.child,
  });

  @override
  State<AppWrapper> createState() => _AppWrapperState();
}

class _AppWrapperState extends State<AppWrapper> {
  @override
  void initState() {
    super.initState();
    // Initialize auth service
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<AuthService>().initialize();
    });
  }

  @override
  Widget build(BuildContext context) {
    return widget.child;
  }
}
