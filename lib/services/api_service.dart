import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../models/detection_result.dart';
import '../models/doctor.dart';

class ApiService extends ChangeNotifier {
  // Try different URLs based on platform
  static String get baseUrl {
    // For development, try localhost first
    return 'http://localhost:5000';
  }
  
  // Alternative URLs to try if localhost fails
  static const List<String> fallbackUrls = [
    'http://127.0.0.1:5000',
    'http://10.0.2.2:5000', // For Android emulator
  ];
  
  bool _isLoading = false;
  String? _error;

  bool get isLoading => _isLoading;
  String? get error => _error;

  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String? error) {
    _error = error;
    notifyListeners();
  }

  Future<DetectionResult?> analyzeImage(File imageFile) async {
    _setLoading(true);
    _setError(null);

    // Try the main URL first
    String? lastError;
    
    try {
      return await _tryAnalyzeImage(imageFile, baseUrl);
    } catch (e) {
      lastError = e.toString();
      print('Failed to connect to $baseUrl: $e');
    }

    // Try fallback URLs
    for (String url in fallbackUrls) {
      try {
        return await _tryAnalyzeImage(imageFile, url);
      } catch (e) {
        lastError = e.toString();
        print('Failed to connect to $url: $e');
      }
    }

    // If all URLs fail, show helpful error message
    _setError('Cannot connect to backend server. Please ensure:\n'
        '1. Backend server is running (python run.py)\n'
        '2. Server is accessible at http://localhost:5000\n'
        '3. No firewall is blocking the connection\n\n'
        'Last error: $lastError');
    return null;
  }

  Future<DetectionResult?> _tryAnalyzeImage(File imageFile, String url) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$url/analyze'),
    );

    request.files.add(
      await http.MultipartFile.fromPath('image', imageFile.path),
    );

    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return DetectionResult.fromJson(data);
    } else {
      throw Exception('Server returned ${response.statusCode}: ${response.body}');
    }
  }

  Future<List<Doctor>> getDoctors() async {
    _setLoading(true);
    _setError(null);

    // Try the main URL first
    try {
      return await _tryGetDoctors(baseUrl);
    } catch (e) {
      print('Failed to connect to $baseUrl: $e');
    }

    // Try fallback URLs
    for (String url in fallbackUrls) {
      try {
        return await _tryGetDoctors(url);
      } catch (e) {
        print('Failed to connect to $url: $e');
      }
    }

    // If all URLs fail, return mock data for demo purposes
    _setError('Cannot connect to backend server. Showing demo data.');
    return _getMockDoctors();
  }

  Future<List<Doctor>> _tryGetDoctors(String url) async {
    final response = await http.get(
      Uri.parse('$url/doctors'),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => Doctor.fromJson(json)).toList();
    } else {
      throw Exception('Server returned ${response.statusCode}: ${response.body}');
    }
  }

  List<Doctor> _getMockDoctors() {
    return [
      Doctor(
        id: '1',
        name: 'Dr. Sarah Johnson',
        specialization: 'Neurology',
        imageUrl: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=face',
        rating: 4.9,
        experience: 15,
        isOnline: true,
        hospital: 'Mayo Clinic',
        languages: ['English', 'Spanish'],
        bio: 'Specialized in facial nerve disorders and stroke rehabilitation.',
      ),
      Doctor(
        id: '2',
        name: 'Dr. Michael Chen',
        specialization: 'Telemedicine',
        imageUrl: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=face',
        rating: 4.8,
        experience: 12,
        isOnline: true,
        hospital: 'Johns Hopkins',
        languages: ['English', 'Mandarin'],
        bio: 'Expert in remote neurological consultations and AI-assisted diagnosis.',
      ),
      Doctor(
        id: '3',
        name: 'Dr. Emily Rodriguez',
        specialization: 'Neurology',
        imageUrl: 'https://images.unsplash.com/photo-1594824373636-4b0b0b0b0b0b?w=300&h=300&fit=crop&crop=face',
        rating: 4.7,
        experience: 10,
        isOnline: false,
        hospital: 'Cleveland Clinic',
        languages: ['English', 'Spanish', 'Portuguese'],
        bio: 'Focused on early detection and treatment of facial paralysis.',
      ),
      Doctor(
        id: '4',
        name: 'Dr. James Wilson',
        specialization: 'Telemedicine',
        imageUrl: 'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=300&h=300&fit=crop&crop=face',
        rating: 4.9,
        experience: 18,
        isOnline: true,
        hospital: 'Massachusetts General',
        languages: ['English'],
        bio: 'Pioneer in telemedicine and digital health solutions.',
      ),
    ];
  }

  Future<bool> sendReportToDoctor(String doctorId, DetectionResult result) async {
    _setLoading(true);
    _setError(null);

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/send-report'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'doctor_id': doctorId,
          'result': result.toJson(),
        }),
      );

      if (response.statusCode == 200) {
        return true;
      } else {
        _setError('Failed to send report: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      _setError('Error sending report: $e');
      return false;
    } finally {
      _setLoading(false);
    }
  }

  Future<bool> submitContactForm({
    required String name,
    required String email,
    required String message,
  }) async {
    _setLoading(true);
    _setError(null);

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/contact'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'name': name,
          'email': email,
          'message': message,
        }),
      );

      if (response.statusCode == 200) {
        return true;
      } else {
        _setError('Failed to submit form: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      _setError('Error submitting form: $e');
      return false;
    } finally {
      _setLoading(false);
    }
  }
}
