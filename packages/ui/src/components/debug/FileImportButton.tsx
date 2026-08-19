import { useRef } from "react";
import { FolderOpen } from "lucide-react";
import { Button } from "@/components/ui/button";

type FileImportButtonProps = {
  accept: string;
  label: string;
  disabled?: boolean;
  onLoad: (text: string) => void;
};

export function FileImportButton({ accept, label, disabled, onLoad }: FileImportButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!file) return;
          void file.text().then(onLoad);
        }}
      />
      <Button
        type="button"
        variant="outline"
        size="icon"
        className="size-6"
        disabled={disabled}
        title={label}
        aria-label={label}
        onClick={() => inputRef.current?.click()}
      >
        <FolderOpen className="h-3.5 w-3.5" />
      </Button>
    </>
  );
}
