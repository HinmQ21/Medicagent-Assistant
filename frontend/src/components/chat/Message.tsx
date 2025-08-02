"use client";

import { Message as MessageType } from '@/types/chat';
import { Card } from '@/components/ui/card';
import { Avatar } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Play, X } from 'lucide-react';
import { useState, useRef } from 'react';
import { generateSpeech } from '@/lib/api';
import { marked } from 'marked';

interface MessageProps {
  message: MessageType;
  onValidation?: (validation: string, comments: string) => Promise<void>;
}

export function Message({ message, onValidation }: MessageProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  // System is now English-only
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [validationComments, setValidationComments] = useState('');

  const handlePlayAudio = async () => {
    if (isGenerating || isPlaying) return;

    try {
      setIsGenerating(true);
      
      // Log the content that's being sent for speech generation
      console.log("Sending for speech generation:", message.content);
      
      // Check if the content is just markdown
      const stripMarkdown = (text: string) => {
        // First, handle basic markdown
        let cleaned = text
          .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
          .replace(/\*(.*?)\*/g, '$1')     // Remove italic
          .replace(/\[(.*?)\]\((.*?)\)/g, '$1') // Remove links
          .replace(/#{1,6}\s?(.*?)$/gm, '$1') // Remove headings
          .replace(/`{3}.*?\n([\s\S]*?)`{3}/g, '$1') // Remove code blocks
          .replace(/`(.*?)`/g, '$1') // Remove inline code
          .replace(/\n{2,}/g, '\n') // Replace multiple newlines with single
          .trim();
          
        // Handle special cases for punctuation formatting
        // Clean up spacing around punctuation marks
        cleaned = cleaned
          .replace(/\s+\,/g, ',') // Remove space before comma
          .replace(/\s+\./g, '.') // Remove space before period
          .replace(/\,\s+/g, ', ') // Ensure space after comma
          .replace(/\.\s+/g, '. ') // Ensure space after period
          .replace(/\s+\?/g, '?') // Remove space before question mark
          .replace(/\?\s+/g, '? ') // Ensure space after question mark
          .replace(/\s+\!/g, '!') // Remove space before exclamation mark
          .replace(/\!\s+/g, '! ') // Ensure space after exclamation mark
          .replace(/\s+\;/g, ';') // Remove space before semicolon
          .replace(/\;\s+/g, '; '); // Ensure space after semicolon
          
        // Check for problematic punctuation-only content
        if (cleaned.trim().length < 2 || /^[,.?!;:]+$/.test(cleaned.trim())) {
          console.warn("Detected punctuation-only content, might be a TTS issue");
        }
          
        return cleaned;
      };
      
      // Get and clean the text
      let cleanedText = stripMarkdown(message.content);
      console.log("Cleaned text:", cleanedText);
      
      // Additional check to prevent lone punctuation
      if (cleanedText.trim().length < 2 || /^[,.?!;:]$/.test(cleanedText.trim())) {
        console.warn("Text too short or just punctuation, adding default message");
        cleanedText = "Sorry, this text cannot be spoken.";
      }
      
      try {
        const audioBlob = await generateSpeech({
          text: cleanedText
        });
        
        // Check if we received a valid audio blob
        if (!audioBlob || audioBlob.size < 100) {
          console.error("Received invalid audio response", audioBlob);
          throw new Error("Invalid audio response");
        }
        
        const audioUrl = URL.createObjectURL(audioBlob);
        if (audioRef.current) {
          audioRef.current.src = audioUrl;
          audioRef.current.playbackRate = 1.2; // Default speed is 1.2x
          
          // Set up error handling for audio playback
          audioRef.current.onerror = (e) => {
            console.error("Audio playback error:", e);
            setIsPlaying(false);
            setIsGenerating(false);
            URL.revokeObjectURL(audioUrl);
          };
          
          // Start playback
          audioRef.current.play()
            .then(() => {
              setIsPlaying(true);
            })
            .catch(err => {
              console.error("Failed to play audio:", err);
              setIsPlaying(false);
              URL.revokeObjectURL(audioUrl);
            });
        }
      } catch (error) {
        console.error('Failed to generate or play speech:', error);
        // Show a user-friendly error message
        alert("Unable to speak this text.");
      }
    } catch (error) {
      console.error('Failed to generate speech:', error);
    } finally {
      setIsGenerating(false);
    }
  };
  
  const handlePauseAudio = () => {
    if (audioRef.current && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
      // Clear the audio source to stop it completely
      if (audioRef.current.src) {
        URL.revokeObjectURL(audioRef.current.src);
        audioRef.current.src = '';
      }
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    if (audioRef.current) {
      URL.revokeObjectURL(audioRef.current.src);
    }
  };

  const handleValidation = async (validation: string) => {
    if (onValidation) {
      await onValidation(validation, validationComments);
    }
  };

  // Language toggle removed - system is English-only

  // Configure marked options for better table rendering
  marked.setOptions({
    breaks: true,
    gfm: true
  });

  return (
    <div className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
      {message.role === 'assistant' && (
        <Avatar className="h-8 w-8">
          <div className="bg-primary text-primary-foreground flex h-full w-full items-center justify-center">
            AI
          </div>
        </Avatar>
      )}
      
      <Card className={`max-w-[80%] p-4 ${message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
        {message.agent && (
          <div className="text-xs font-semibold mb-2">{message.agent}</div>
        )}
        
        <div className="prose prose-sm dark:prose-invert max-w-none overflow-x-auto" 
             dangerouslySetInnerHTML={{ 
               __html: marked(message.content) 
             }} />
        
        {/* Custom styles for tables */}
        <style jsx global>{`
          .prose table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.875rem;
          }
          .prose th {
            background-color: rgba(0, 0, 0, 0.1);
            font-weight: 600;
            text-align: left;
            padding: 0.75rem;
            border: 1px solid rgba(0, 0, 0, 0.1);
          }
          .prose td {
            padding: 0.75rem;
            border: 1px solid rgba(0, 0, 0, 0.1);
          }
          .prose tr:nth-child(even) {
            background-color: rgba(0, 0, 0, 0.05);
          }
          .prose img {
            max-width: 100%;
            height: auto;
            border-radius: 0.5rem;
            margin: 1rem 0;
          }
          .prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
          }
          .prose ul, .prose ol {
            margin: 1rem 0;
            padding-left: 1.5rem;
          }
          .prose li {
            margin: 0.5rem 0;
          }
          .prose blockquote {
            border-left: 4px solid rgba(0, 0, 0, 0.1);
            padding-left: 1rem;
            margin: 1rem 0;
            font-style: italic;
          }
          .prose code {
            background-color: rgba(0, 0, 0, 0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.875em;
          }
          .prose pre {
            background-color: rgba(0, 0, 0, 0.1);
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
          }
          .prose a {
            color: #2563eb;
            text-decoration: underline;
          }
          .prose a:hover {
            color: #1d4ed8;
          }
        `}</style>
        
        {message.image && (
          <div className="mt-2">
            <img 
              src={message.image} 
              alt="Image" 
              className="max-w-full rounded" 
              style={{ maxHeight: '300px', objectFit: 'contain' }}
            />
          </div>
        )}
        
        {message.resultImage && (
          <div className="mt-2">
            <img 
              src={message.resultImage} 
              alt="Result Image" 
              className="max-w-full rounded" 
              style={{ maxHeight: '300px', objectFit: 'contain' }}
            />
          </div>
        )}
        
        {message.role === 'assistant' && (
          <div className="mt-2 flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePlayAudio}
              disabled={isGenerating || isPlaying}
              title="Play"
            >
              {isGenerating ? (
                <div className="animate-spin">⌛</div>
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
            {isPlaying && (
              <Button
                variant="outline"
                size="sm"
                onClick={handlePauseAudio}
                title="Stop"
                className="bg-red-50 hover:bg-red-100 border-red-200 text-red-500"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
            {/* Language toggle removed - system is English-only */}
          </div>
        )}
        
        {message.agent?.includes('HUMAN_VALIDATION') && onValidation && (
          <div className="mt-4 space-y-2">
            <div className="text-sm font-medium">Do you agree with this result?</div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleValidation('yes')}
              >
                Yes
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleValidation('no')}
              >
                No
              </Button>
            </div>
            <textarea
              className="w-full p-2 text-sm border rounded"
              placeholder="Add comments (optional)"
              value={validationComments}
              onChange={(e) => setValidationComments(e.target.value)}
            />
          </div>
        )}
      </Card>
      
      {message.role === 'user' && (
        <Avatar className="h-8 w-8">
          <div className="bg-secondary text-secondary-foreground flex h-full w-full items-center justify-center">
            U
          </div>
        </Avatar>
      )}
      
      <audio
        ref={audioRef}
        onEnded={handleAudioEnded}
        onPause={() => setIsPlaying(false)}
        className="hidden"
      />
    </div>
  );
}