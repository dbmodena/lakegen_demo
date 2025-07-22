import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Minus } from "lucide-react";

export default function NumberInput() {
  // A helper function to handle value changes
  const handleChange = (newValue) => {
    // Ensure the new value is a valid number
    const numericValue = parseFloat(newValue);
    if (isNaN(numericValue)) return;

    // 1. Update the UI immediately
    updateElement({ ...props, value: numericValue });

    // 2. Send the new value to the Python backend
    callAction({
      name: "on_number_change",
      payload: { value: numericValue },
    });
  };

  const step = props.step || 1;
  const value = props.value || 0;

  return (
    <div className="p-4 w-full max-w-xs space-y-2">
      {/* The label is set from Python props */}
      {props.label && <Label>{props.label}</Label>}

      <div className="flex items-center gap-2">
        {/* Decrement Button */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => handleChange(value - step)}
        >
          <Minus className="h-4 w-4" />
        </Button>

        {/* Number Input Field */}
        <Input
          type="number"
          value={value}
          onChange={(e) => handleChange(e.target.value)}
          className="text-center"
          step={step}
        />

        {/* Increment Button */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => handleChange(value + step)}
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}