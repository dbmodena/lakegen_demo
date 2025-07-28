import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function Dropdown() {
  const handleSelectChange = (newValue) => {
    updateElement({ ...props, value: newValue });
    callAction({
      name: "on_select",
      payload: { value: newValue },
    });
  };

  return (
    <div className="w-full max-w-xs p-4">
      {/* Change the hardcoded title to render props.title */}
      {props.title && (
        <h4 className="text-lg font-semibold mb-2">{props.title}</h4>
      )}
      
      <Select onValueChange={handleSelectChange} value={props.value}>
        <SelectTrigger>
          <SelectValue placeholder="Select an option..." />
        </SelectTrigger>
        <SelectContent>
          {props.options?.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}