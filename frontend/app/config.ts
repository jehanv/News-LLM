export const config = {
    // Use environment variable in production, fallback to localhost in development
    apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'
};
