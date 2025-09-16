import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

class ImageService extends ChangeNotifier {
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  List<CameraDescription>? _cameras;
  CameraController? _cameraController;
  bool _isCameraInitialized = false;

  File? get selectedImage => _selectedImage;
  List<CameraDescription>? get cameras => _cameras;
  CameraController? get cameraController => _cameraController;
  bool get isCameraInitialized => _isCameraInitialized;

  Future<bool> initializeCamera() async {
    try {
      // Check camera permission
      final status = await Permission.camera.status;
      if (status != PermissionStatus.granted) {
        final result = await Permission.camera.request();
        if (result != PermissionStatus.granted) {
          debugPrint('Camera permission denied');
          return false;
        }
      }

      _cameras = await availableCameras();
      if (_cameras!.isNotEmpty) {
        _cameraController = CameraController(
          _cameras![0],
          ResolutionPreset.high,
          enableAudio: false,
        );
        await _cameraController!.initialize();
        _isCameraInitialized = true;
        notifyListeners();
        return true;
      }
      return false;
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      return false;
    }
  }

  Future<void> pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      
      if (image != null) {
        _selectedImage = File(image.path);
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Error picking image from gallery: $e');
    }
  }

  Future<void> captureImageFromCamera() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final XFile image = await _cameraController!.takePicture();
      _selectedImage = File(image.path);
      notifyListeners();
    } catch (e) {
      debugPrint('Error capturing image: $e');
    }
  }

  void clearSelectedImage() {
    _selectedImage = null;
    notifyListeners();
  }

  void disposeCamera() {
    _cameraController?.dispose();
    _cameraController = null;
    _isCameraInitialized = false;
    notifyListeners();
  }

  @override
  void dispose() {
    disposeCamera();
    super.dispose();
  }
}
