class User {
  final String id;
  final String email;
  final String name;
  final String? phone;
  final DateTime createdAt;
  final DateTime? lastLogin;
  final UserRole role;

  User({
    required this.id,
    required this.email,
    required this.name,
    this.phone,
    required this.createdAt,
    this.lastLogin,
    this.role = UserRole.patient,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] ?? '',
      email: json['email'] ?? '',
      name: json['name'] ?? '',
      phone: json['phone'],
      createdAt: DateTime.parse(json['created_at'] ?? DateTime.now().toIso8601String()),
      lastLogin: json['last_login'] != null ? DateTime.parse(json['last_login']) : null,
      role: UserRole.values.firstWhere(
        (e) => e.toString() == 'UserRole.${json['role']}',
        orElse: () => UserRole.patient,
      ),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'name': name,
      'phone': phone,
      'created_at': createdAt.toIso8601String(),
      'last_login': lastLogin?.toIso8601String(),
      'role': role.toString().split('.').last,
    };
  }

  User copyWith({
    String? id,
    String? email,
    String? name,
    String? phone,
    DateTime? createdAt,
    DateTime? lastLogin,
    UserRole? role,
  }) {
    return User(
      id: id ?? this.id,
      email: email ?? this.email,
      name: name ?? this.name,
      phone: phone ?? this.phone,
      createdAt: createdAt ?? this.createdAt,
      lastLogin: lastLogin ?? this.lastLogin,
      role: role ?? this.role,
    );
  }
}

enum UserRole {
  patient,
  doctor,
  admin,
}
