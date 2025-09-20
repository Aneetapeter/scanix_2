import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

class ImageService extends ChangeNotifier {
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  Uint8List? _selectedImageBytes;
  List<CameraDescription>? _cameras;
  CameraController? _cameraController;
  bool _isCameraInitialized = false;

  File? get selectedImage => _selectedImage;
  Uint8List? get selectedImageBytes => _selectedImageBytes;
  List<CameraDescription>? get cameras => _cameras;
  CameraController? get cameraController => _cameraController;
  bool get isCameraInitialized => _isCameraInitialized;

  Future<bool> initializeCamera() async {
    // Camera is not supported on web
    if (kIsWeb) {
      debugPrint('Camera not supported on web');
      return false;
    }

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
        if (kIsWeb) {
          // For web, read the image as bytes
          final bytes = await image.readAsBytes();
          _selectedImageBytes = Uint8List.fromList(bytes);
          _selectedImage = null;
        } else {
          // For mobile/desktop, use File
          _selectedImage = File(image.path);
          _selectedImageBytes = null;
        }
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Error picking image from gallery: $e');
    }
  }

  Future<void> captureImageFromCamera() async {
    // Camera is not supported on web
    if (kIsWeb) {
      debugPrint('Camera capture not supported on web');
      return;
    }

    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final XFile image = await _cameraController!.takePicture();
      // For mobile/desktop, use File
      _selectedImage = File(image.path);
      _selectedImageBytes = null;
      notifyListeners();
    } catch (e) {
      debugPrint('Error capturing image: $e');
    }
  }

  void clearSelectedImage() {
    _selectedImage = null;
    _selectedImageBytes = null;
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
