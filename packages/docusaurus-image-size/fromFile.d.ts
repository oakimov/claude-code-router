export interface ImageDimensions {
  width: number;
  height: number;
  type?: string;
  orientation?: number;
  images?: ImageDimensions[];
}

export declare function imageSizeFromFile(
  filePath: string
): Promise<ImageDimensions>;

export declare function setConcurrency(value: number): void;
