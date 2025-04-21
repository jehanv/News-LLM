import { NextResponse } from 'next/server';
import { config } from '../../config';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { query } = body;

    if (!query) {
      return NextResponse.json(
        { error: 'Query parameter is required' },
        { status: 400 }
      );
    }

    console.log('Making request to backend:', `${config.apiUrl}/api/search`);
    
    const response = await fetch(`${config.apiUrl}/api/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend error details:', {
        status: response.status,
        statusText: response.statusText,
        url: response.url,
        error: errorText
      });

      // Try to parse error text as JSON
      try {
        const errorJson = JSON.parse(errorText);
        throw new Error(errorJson.detail || errorJson.message || `Backend error: ${response.status}`);
      } catch (parseError) {
        throw new Error(`Backend error (${response.status}): ${errorText}`);
      }
    }

    const data = await response.json();
    
    if (!data || typeof data !== 'object') {
      console.error('Invalid response format:', data);
      throw new Error('Invalid response format from backend');
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in news API route:', error);
    
    // Check if the error is a connection error
    if (error instanceof Error && error.message.includes('fetch')) {
      return NextResponse.json(
        { error: 'Unable to connect to the backend server. It may be starting up, please try again in a moment.' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'An unexpected error occurred' },
      { status: 500 }
    );
  }
}
