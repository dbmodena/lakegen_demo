import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Eye, EyeOff, KeyRound } from "lucide-react";

export default function PasswordInput() {
  // Local state to hold the input value and visibility
  const [value, setValue] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Function to handle the form submission
  const handleSubmit = () => {
    // Send the API key to the Python backend
    callAction({
      name: "submit_api_key",
      payload: { key: value },
    });
    // Update the element to show it has been submitted
    updateElement({ ...props, submitted: true });
  };
  
  // If props.submitted is true, show a confirmation message
  if (props.submitted) {
    return (
        <div className="p-4 w-full max-w-xs text-center text-sm text-green-600">
            API Key submitted successfully.
        </div>
    );
  }

  return (
    <div className="p-4 w-full max-w-xs space-y-2">
      <Label>{props.label || 'Password'}</Label>
      <div className="flex items-center gap-2">
        <div className="relative flex-grow">
          <Input
            type={showPassword ? "text" : "password"}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="••••••••••••••••"
            className="pr-10" // Add padding to the right for the icon
          />
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
            onClick={() => setShowPassword((prev) => !prev)}
          >
            {showPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
      <Button onClick={handleSubmit} className="w-full">
        <KeyRound className="mr-2 h-4 w-4" />
        Submit
      </Button>
    </div>
  );
}