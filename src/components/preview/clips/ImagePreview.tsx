import React from 'react'
import {Image} from 'react-konva'
import { ImageClipProps } from '@/lib/types'

const ImagePreview: React.FC<ImageClipProps & {rectWidth: number, rectHeight: number}> = ({clipId, src, type, startFrame, endFrame, framesToGiveEnd, framesToGiveStart, height, width, rectWidth, rectHeight}) => {
  return (
    <Image image={undefined} width={rectWidth} height={rectHeight} />
  )
}

export default ImagePreview