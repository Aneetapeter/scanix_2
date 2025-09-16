class Doctor {
  final String id;
  final String name;
  final String specialization;
  final String imageUrl;
  final double rating;
  final int experience;
  final bool isOnline;
  final String hospital;
  final List<String> languages;
  final String bio;

  Doctor({
    required this.id,
    required this.name,
    required this.specialization,
    required this.imageUrl,
    required this.rating,
    required this.experience,
    required this.isOnline,
    required this.hospital,
    required this.languages,
    required this.bio,
  });

  factory Doctor.fromJson(Map<String, dynamic> json) {
    return Doctor(
      id: json['id'] ?? '',
      name: json['name'] ?? '',
      specialization: json['specialization'] ?? '',
      imageUrl: json['image_url'] ?? '',
      rating: (json['rating'] ?? 0.0).toDouble(),
      experience: json['experience'] ?? 0,
      isOnline: json['is_online'] ?? false,
      hospital: json['hospital'] ?? '',
      languages: List<String>.from(json['languages'] ?? []),
      bio: json['bio'] ?? '',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'specialization': specialization,
      'image_url': imageUrl,
      'rating': rating,
      'experience': experience,
      'is_online': isOnline,
      'hospital': hospital,
      'languages': languages,
      'bio': bio,
    };
  }
}
