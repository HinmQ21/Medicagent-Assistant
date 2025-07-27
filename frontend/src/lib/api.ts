import { ChatResponse, ValidationResponse, SpeechRequest } from '@/types/chat';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: message,
      conversation_history: [],
    }),
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error('Failed to send message');
  }

  return response.json();
}

export async function uploadImage(image: File, text: string = ''): Promise<ChatResponse> {
  const formData = new FormData();
  formData.append('image', image);
  formData.append('text', text);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error('Failed to upload image');
  }

  const data = await response.json();
  
  // Ensure result_image has the full URL if it exists
  if (data.result_image && !data.result_image.startsWith('http')) {
    data.result_image = `${API_BASE_URL}${data.result_image}`;
  }

  return data;
}

export async function sendValidation(validation: string, comments: string): Promise<ValidationResponse> {
  const formData = new FormData();
  formData.append('validation_result', validation);
  formData.append('comments', comments);

  const response = await fetch(`${API_BASE_URL}/api/validate`, {
    method: 'POST',
    body: formData,
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error('Failed to send validation');
  }

  return response.json();
}

export async function transcribeAudio(audio: Blob, language: string = "vi-VN"): Promise<{ transcript: string }> {
  console.log(`Preparing to transcribe audio: ${audio.size} bytes, language: ${language}`);
  
  const formData = new FormData();
  formData.append('audio', audio);
  formData.append('language', language);

  console.log(`Sending request to ${API_BASE_URL}/api/transcribe`);
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
      method: 'POST',
      body: formData,
    });

    console.log(`Transcription response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Transcription failed: ${response.status}`, errorText);
      throw new Error(`Failed to transcribe audio: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    console.log(`Transcription result:`, data);
    return data;
  } catch (error) {
    console.error('Network or parsing error during transcription:', error);
    throw error;
  }
}

export async function generateSpeech(request: SpeechRequest): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/generate-speech`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error('Failed to generate speech');
  }

  return response.blob();
} 