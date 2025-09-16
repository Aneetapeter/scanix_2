import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../models/doctor.dart';
import '../widgets/navigation_bar.dart';
import '../widgets/footer_section.dart';
import '../widgets/doctor_card.dart';

class DoctorsScreen extends StatefulWidget {
  const DoctorsScreen({super.key});

  @override
  State<DoctorsScreen> createState() => _DoctorsScreenState();
}

class _DoctorsScreenState extends State<DoctorsScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ApiService>().getDoctors();
    });
  }

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
            'Our Medical Team',
            style: TextStyle(
              fontSize: 36,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Connect with experienced neurologists and specialists',
            style: TextStyle(fontSize: 20, color: Color(0xFF6B7280)),
          ),
          const SizedBox(height: 40),

          // Doctors List
          Consumer<ApiService>(
            builder: (context, apiService, child) {
              if (apiService.isLoading) {
                return const Center(
                  child: Padding(
                    padding: EdgeInsets.all(40),
                    child: CircularProgressIndicator(),
                  ),
                );
              }

              if (apiService.error != null) {
                return Center(
                  child: Padding(
                    padding: const EdgeInsets.all(40),
                    child: Column(
                      children: [
                        const Icon(
                          Icons.error_outline,
                          size: 64,
                          color: Colors.red,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Error loading doctors: ${apiService.error}',
                          style: const TextStyle(color: Colors.red),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: () {
                            context.read<ApiService>().getDoctors();
                          },
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  ),
                );
              }

              // Mock data for demonstration
              final doctors = _getMockDoctors();

              return Column(
                children: [
                  // Filter options
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          decoration: InputDecoration(
                            hintText: 'Search doctors...',
                            prefixIcon: const Icon(Icons.search),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      DropdownButton<String>(
                        value: 'All',
                        items: const [
                          DropdownMenuItem(
                            value: 'All',
                            child: Text('All Specialties'),
                          ),
                          DropdownMenuItem(
                            value: 'Neurology',
                            child: Text('Neurology'),
                          ),
                          DropdownMenuItem(
                            value: 'Telemedicine',
                            child: Text('Telemedicine'),
                          ),
                        ],
                        onChanged: (value) {},
                      ),
                    ],
                  ),

                  const SizedBox(height: 32),

                  // Doctors Grid
                  GridView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 0.8,
                          crossAxisSpacing: 16,
                          mainAxisSpacing: 16,
                        ),
                    itemCount: doctors.length,
                    itemBuilder: (context, index) {
                      return DoctorCard(doctor: doctors[index]);
                    },
                  ),
                ],
              );
            },
          ),

          const SizedBox(height: 40),

          // Telemedicine Info
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFFF0F9FF),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFF0EA5E9)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(
                  children: [
                    Icon(Icons.video_call, color: Color(0xFF0EA5E9)),
                    SizedBox(width: 12),
                    Text(
                      'Telemedicine Services',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF0C4A6E),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                const Text(
                  'Connect with our specialists through secure video consultations. Get expert medical advice from the comfort of your home.',
                  style: TextStyle(fontSize: 16, color: Color(0xFF0C4A6E)),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    ElevatedButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.video_call),
                      label: const Text('Start Video Call'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF0EA5E9),
                        foregroundColor: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 12),
                    OutlinedButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.chat),
                      label: const Text('Chat'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: const Color(0xFF0EA5E9),
                        side: const BorderSide(color: Color(0xFF0EA5E9)),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  List<Doctor> _getMockDoctors() {
    return [
      Doctor(
        id: '1',
        name: 'Dr. Sarah Johnson',
        specialization: 'Neurology',
        imageUrl:
            'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=face',
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
        imageUrl:
            'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=face',
        rating: 4.8,
        experience: 12,
        isOnline: true,
        hospital: 'Johns Hopkins',
        languages: ['English', 'Mandarin'],
        bio:
            'Expert in remote neurological consultations and AI-assisted diagnosis.',
      ),
      Doctor(
        id: '3',
        name: 'Dr. Emily Rodriguez',
        specialization: 'Neurology',
        imageUrl:
            'https://images.unsplash.com/photo-1594824373636-4b0b0b0b0b0b?w=300&h=300&fit=crop&crop=face',
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
        imageUrl:
            'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=300&h=300&fit=crop&crop=face',
        rating: 4.9,
        experience: 18,
        isOnline: true,
        hospital: 'Massachusetts General',
        languages: ['English'],
        bio: 'Pioneer in telemedicine and digital health solutions.',
      ),
    ];
  }
}
