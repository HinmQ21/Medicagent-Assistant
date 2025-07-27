"use client";

import { useState, useRef, useEffect } from 'react';
import { Message as MessageType } from '@/types/chat';
import { Message } from './Message';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { Mic, MicOff, Paperclip, Send, Trash2 } from 'lucide-react';
import { sendChatMessage, uploadImage, sendValidation, transcribeAudio } from '@/lib/api';

export function Chat() {
  const [messages, setMessages] = useState<MessageType[]>([
    {
      role: 'assistant',
      content: "Chào bạn, tôi là hệ thống Medicagent. Tôi có chuyên môn trong 3 lĩnh vực chính: Khối u não, X-quang ngực, và phân vùng tổn thương da. Tuy nhiên, tôi chỉ đóng vai trò hỗ trợ và không thể thay thế được các chuyên gia y tế. Nếu có câu hỏi hoặc cần chẩn đoán, hãy gửi thông tin nhé, tôi sẽ giúp bạn tìm kiếm thông tin chính xác nhất!",
    },
  ]);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if ((!input.trim() && !selectedImage) || isProcessing) return;

    const newMessage: MessageType = {
      role: 'user',
      content: input.trim(),
      image: selectedImage || undefined,
    };

    setMessages((prev) => [...prev, newMessage]);
    setInput('');
    setSelectedImage(null);
    setIsProcessing(true);

    try {
      let response;
      if (selectedImage) {
        const imageFile = await fetch(selectedImage).then((r) => r.blob());
        response = await uploadImage(new File([imageFile], 'image.jpg'), input.trim());
      } else {
        response = await sendChatMessage(input.trim());
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          agent: response.agent,
          resultImage: response.result_image,
        },
      ]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: 'Sorry, there was an error processing your request. Please try again.',
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleValidation = async (validation: string, comments: string) => {
    setIsProcessing(true);
    try {
      const response = await sendValidation(validation, comments);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          agent: 'HUMAN_VALIDATED',
        },
      ]);
    } catch (error) {
      console.error('Error sending validation:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: 'Sorry, there was an error processing your validation. Please try again.',
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        try {
          const { transcript } = await transcribeAudio(audioBlob);
          setInput(transcript);
        } catch (error) {
          console.error('Error transcribing audio:', error);
        }
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'system',
        content: "Xin chào, hãy tiếp tục với Medicagent nhé!",
      },
    ]);
  };

  return (
    <div className="flex h-screen bg-[#F4F6F8] overflow-hidden">
      {/* Sidebar */}
      <Card className="w-72 h-full py-3 px-3 hidden md:flex md:flex-col border-[#2A9DF4] border-r border-l-0 border-t-0 border-b-0 rounded-none bg-white shadow-sm">
        <h2 className="text-lg font-semibold mb-3 flex items-center gap-2 text-[#2A9DF4] px-2">
          <i className="fas fa-robot" />
          Medicagent
        </h2>
        <div className="space-y-3 flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-[#2A9DF4] scrollbar-track-transparent">
          {/* Tác vụ hội thoại */}
          <div className="bg-[#F4F6F8]/50 p-2 rounded-md">
            <h3 className="font-medium mb-1 flex items-center gap-2 text-[#2A9DF4] text-sm">
              <i className="fas fa-comment-medical" />
              Tác vụ hội thoại
            </h3>
            <ul className="text-xs space-y-1">
              <li className="flex flex-col gap-1 hover:text-[#2A9DF4] transition-colors p-1">
                <div className="flex items-center gap-2">
                  <i className="fas fa-comment" />
                  Conversation Agent
                </div>
                <div className="pl-5 text-[10px] text-gray-600">
                  <p>Hội thoại chung</p>
                </div>
              </li>
              <li className="flex flex-col gap-1 hover:text-[#2A9DF4] transition-colors p-1">
                <div className="flex items-center gap-2">
                  <i className="fas fa-database" />
                  Medical RAG Agent
                </div>
                <div className="pl-5 text-[10px] text-gray-600">
                  <p>Truy xuất thông tin y khoa: </p>
                  <p>• Phân tích PDF dựa trên Docling</p>
                  <p>• Nhúng nội dung định dạng markdown</p>
                  <p>• Phân đoạn ngữ nghĩa dựa trên LLM</p>
                  <p>• Tìm kiếm lai Qdrant Vector DB</p>
                </div>
              </li>
              <li className="flex flex-col gap-1 hover:text-[#2A9DF4] transition-colors p-1">
                <div className="flex items-center gap-2">
                  <i className="fas fa-search" />
                  Web Search Agent
                </div>
                <div className="pl-5 text-[10px] text-gray-600">
                  <p>Tìm kiếm thông tin: </p>
                  <p>• PubMed Search: nghiên cứu y học</p>
                  <p>• Tavily Search: tìm kiếm đa nguồn</p>
                </div>
              </li>
            </ul>
          </div>

          {/* Tác vụ thị giác máy tính */}
          <div className="bg-[#F4F6F8]/50 p-2 rounded-md">
            <h3 className="font-medium mb-1 flex items-center gap-2 text-[#2A9DF4] text-sm">
              <i className="fas fa-camera" />
              Tác vụ thị giác máy tính
            </h3>
            <ul className="text-xs space-y-1">
              <li className="flex flex-col gap-1 hover:text-[#2A9DF4] transition-colors p-1">
                <div className="flex items-center gap-2">
                  <i className="fas fa-brain" />
                  Brain Tumor Agent
                </div>
                <div className="pl-5 text-[10px] text-gray-600">
                  <p>• Phân loại hình ảnh MRI não</p>
                  <p>• Accuracy: 97.56%</p>
                </div>
              </li>
              <li className="flex flex-col gap-1 hover:text-[#2A9DF4] transition-colors p-1">
                <div className="flex items-center gap-2">
                  <i className="fas fa-lungs" />
                  Chest Xray Agent
                </div>
                <div className="pl-5 text-[10px] text-gray-600">
                  <p>• Nhận diện Covid-19 từ X-quang</p>
                  <p>• Accuracy: 97%</p>
                </div>
              </li>
              <li className="flex flex-col gap-1 hover:text-[#2A9DF4] transition-colors p-1">
                <div className="flex items-center gap-2">
                  <i className="fas fa-allergies" />
                  Skin Lesion Agent
                </div>
                <div className="pl-5 text-[10px] text-gray-600">
                  <p>• Phân vùng tổn thương da</p>
                  <p>• Dice Score: 0.784</p>
                </div>
              </li>
            </ul>
          </div>
        </div>

        <Button
          className="w-full mt-2 bg-[#D9534F] hover:bg-[#F44336] text-white"
          onClick={clearChat}
        >
          <Trash2 className="h-3.5 w-3.5 mr-2" />
          Xóa Cuộc Hội Thoại
        </Button>
      </Card>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white h-full">
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 bg-gradient-to-b from-white to-[#F4F6F8]"
        >
          {messages.map((message, index) => (
            <Message
              key={index}
              message={message}
              onValidation={message.agent?.includes('HUMAN_VALIDATION') ? handleValidation : undefined}
            />
          ))}
          {isProcessing && (
            <div className="flex justify-center">
              <div className="animate-pulse flex space-x-2">
                <div className="h-2 w-2 bg-[#2A9DF4] rounded-full"></div>
                <div className="h-2 w-2 bg-[#2A9DF4] rounded-full"></div>
                <div className="h-2 w-2 bg-[#2A9DF4] rounded-full"></div>
              </div>
            </div>
          )}
        </div>

        <div className="p-3 border-t border-[#F4F6F8] bg-white">
        {selectedImage && (
        <div className="mb-2 relative inline-block">
          <img
            src={selectedImage}
            alt="Preview"
            className="h-16 w-16 object-cover rounded-md shadow-sm border border-gray-300"
          />
          <button
            className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-md hover:bg-red-600 transition-colors"
            onClick={() => setSelectedImage(null)}
          >
            ×
          </button>
        </div>
      )}
          <div className="flex gap-2 items-center">
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleImageUpload}
            />
            <Button
              variant="outline"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              className="text-[#2A9DF4] border-[#2A9DF4] hover:bg-[#2A9DF4] hover:text-white transition-all h-9 w-9"
            >
              <Paperclip className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={toggleRecording}
              className={isRecording 
                ? 'bg-[#A3D9A5] text-white border-[#A3D9A5] hover:bg-[#8FCC91] h-9 w-9' 
                : 'text-[#2A9DF4] border-[#2A9DF4] hover:bg-[#2A9DF4] hover:text-white transition-all h-9 w-9'}
            >
              {isRecording ? (
                <MicOff className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )}
            </Button>
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Nhập câu hỏi..."
              className="flex-1 border-[#2A9DF4] focus-visible:ring-[#2A9DF4] rounded-md resize-none h-9 min-h-[36px] py-2 px-3"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <Button
              onClick={handleSendMessage}
              disabled={isProcessing || (!input.trim() && !selectedImage)}
              className="bg-[#2A9DF4] hover:bg-[#1E8CE3] text-white h-9"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}