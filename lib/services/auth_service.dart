import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user.dart';

class AuthService extends ChangeNotifier {
  User? _currentUser;
  bool _isLoading = false;
  String? _error;

  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get isAuthenticated => _currentUser != null;

  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String? error) {
    _error = error;
    notifyListeners();
  }

  Future<void> initialize() async {
    _setLoading(true);
    try {
      final prefs = await SharedPreferences.getInstance();

      // Initialize demo data if not exists
      await _initializeDemoData(prefs);

      final userJson = prefs.getString('current_user');
      if (userJson != null) {
        final userData = json.decode(userJson);
        _currentUser = User.fromJson(userData);
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('Error initializing auth: $e');
      }
    } finally {
      _setLoading(false);
    }
  }

  Future<void> _initializeDemoData(SharedPreferences prefs) async {
    final usersJson = prefs.getString('users');
    if (usersJson == null) {
      // Create demo users
      final demoUsers = [
        {
          'id': 'demo1',
          'email': 'demo@scanix.ai',
          'password': 'demo123',
          'name': 'Demo User',
          'phone': '+1 (555) 123-4567',
          'created_at': DateTime.now().toIso8601String(),
          'role': 'patient',
        },
        {
          'id': 'demo2',
          'email': 'doctor@scanix.ai',
          'password': 'doctor123',
          'name': 'Dr. Sarah Johnson',
          'phone': '+1 (555) 987-6543',
          'created_at': DateTime.now().toIso8601String(),
          'role': 'doctor',
        },
      ];

      await prefs.setString('users', json.encode(demoUsers));
    }
  }

  Future<bool> signUp({
    required String email,
    required String password,
    required String name,
    String? phone,
    UserRole role = UserRole.patient,
  }) async {
    _setLoading(true);
    _setError(null);

    try {
      // Simulate API call delay
      await Future.delayed(const Duration(seconds: 1));

      // Validate input
      if (email.isEmpty || password.isEmpty || name.isEmpty) {
        _setError('All fields are required');
        return false;
      }

      if (!_isValidEmail(email)) {
        _setError('Please enter a valid email address');
        return false;
      }

      if (password.length < 6) {
        _setError('Password must be at least 6 characters');
        return false;
      }

      // Create new user
      final user = User(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        email: email,
        name: name,
        phone: phone,
        createdAt: DateTime.now(),
        role: role,
      );

      // Save to local storage with password
      await _saveUserWithPassword(user, password);
      _currentUser = user;
      notifyListeners();

      return true;
    } catch (e) {
      _setError('Sign up failed: $e');
      return false;
    } finally {
      _setLoading(false);
    }
  }

  Future<bool> signIn({required String email, required String password}) async {
    _setLoading(true);
    _setError(null);

    try {
      // Simulate API call delay
      await Future.delayed(const Duration(seconds: 1));

      // Validate input
      if (email.isEmpty || password.isEmpty) {
        _setError('Email and password are required');
        return false;
      }

      // Check if user exists in local storage
      final prefs = await SharedPreferences.getInstance();
      final usersJson = prefs.getString('users') ?? '[]';
      final List<dynamic> usersData = json.decode(usersJson);

      final userData = usersData.firstWhere(
        (user) => user['email'] == email && user['password'] == password,
        orElse: () => null,
      );

      if (userData == null) {
        _setError('Invalid email or password');
        return false;
      }

      // Create user object (without password)
      final user = User.fromJson(userData);
      user.copyWith(lastLogin: DateTime.now());

      // Save current user
      await _saveUser(user);
      _currentUser = user;
      notifyListeners();

      return true;
    } catch (e) {
      _setError('Sign in failed: $e');
      return false;
    } finally {
      _setLoading(false);
    }
  }

  Future<void> signOut() async {
    _setLoading(true);
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove('current_user');
      _currentUser = null;
      notifyListeners();
    } catch (e) {
      if (kDebugMode) {
        debugPrint('Error signing out: $e');
      }
    } finally {
      _setLoading(false);
    }
  }

  Future<void> _saveUser(User user) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('current_user', json.encode(user.toJson()));
  }

  Future<void> _saveUserWithPassword(User user, String password) async {
    final prefs = await SharedPreferences.getInstance();

    // Save current user
    await prefs.setString('current_user', json.encode(user.toJson()));

    // Save to users list with password
    final usersJson = prefs.getString('users') ?? '[]';
    final List<dynamic> usersData = json.decode(usersJson);

    // Remove existing user with same email
    usersData.removeWhere((u) => u['email'] == user.email);

    // Add new user
    final userData = user.toJson();
    userData['password'] = password;
    usersData.add(userData);

    await prefs.setString('users', json.encode(usersData));
  }

  bool _isValidEmail(String email) {
    return RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$').hasMatch(email);
  }

  Future<void> updateProfile({String? name, String? phone}) async {
    if (_currentUser == null) return;

    _setLoading(true);
    try {
      _currentUser = _currentUser!.copyWith(
        name: name ?? _currentUser!.name,
        phone: phone ?? _currentUser!.phone,
      );

      await _saveUser(_currentUser!);
      notifyListeners();
    } catch (e) {
      _setError('Failed to update profile: $e');
    } finally {
      _setLoading(false);
    }
  }
}
