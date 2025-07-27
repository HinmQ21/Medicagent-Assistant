export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  image?: string;
  agent?: string;
  resultImage?: string;
}

export interface ChatResponse {
  status: string;
  response: string;
  agent: string;
  result_image?: string;
}

export interface ValidationResponse {
  status: string;
  message: string;
  response: string;
}

export interface SpeechRequest {
  text: string;
  voice_id?: string;
  language?: string;
} 