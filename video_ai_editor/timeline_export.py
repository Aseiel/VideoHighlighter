import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

class TimelineExporter:
    """Export edit timeline to various formats"""
    
    @staticmethod
    def to_edl(clips, video_path, output_path=None, fps=30):
        """Export to CMX3600 EDL - DaVinci Resolve compatible"""
        
        def seconds_to_timecode(seconds, fps=30, drop_frame=False):
            """Convert seconds to SMPTE timecode"""
            total_frames = int(round(seconds * fps))
            hours = total_frames // (3600 * fps)
            minutes = (total_frames // (60 * fps)) % 60
            secs = (total_frames // fps) % 60
            frames = total_frames % fps
            return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
        
        lines = []
        
        # Header
        lines.append("TITLE: AI Video Editor Edit")
        lines.append("FCM: NON-DROP FRAME")
        lines.append("")
        
        # Reel name from filename (without extension)
        reel_name = os.path.splitext(os.path.basename(video_path))[0]
        # Limit reel name to 8 chars for compatibility
        reel_name = reel_name[:8].upper()
        
        # Add source file reference
        lines.append(f"* SOURCE FILE: {video_path}")
        lines.append("")
        
        # Each clip
        for i, (start, end) in enumerate(clips, 1):
            duration = end - start
            
            # Calculate cumulative time for record track
            record_start = sum(clips[j][1] - clips[j][0] for j in range(i-1))
            record_end = record_start + duration
            
            # Convert to timecode
            source_in = seconds_to_timecode(start, fps)
            source_out = seconds_to_timecode(end, fps)
            record_in = seconds_to_timecode(record_start, fps)
            record_out = seconds_to_timecode(record_end, fps)
            
            # EDL entry - proper format for DaVinci Resolve
            lines.append(f"{i:03d}  {reel_name:8} V     C        {source_in} {source_out} {record_in} {record_out}")
            lines.append(f"* FROM CLIP NAME: {os.path.basename(video_path)}")
            lines.append(f"* COMMENT: Clip {i} - {duration:.1f}s")
            lines.append("")
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    @staticmethod
    def to_fcp_xml(clips, video_path, output_path=None, fps=30):
        """
        Export to Final Cut Pro XML format (DaVinci Resolve compatible)
        
        Args:
            clips: List of (start_time, end_time) tuples in seconds
            video_path: Path to source video
            output_path: Output file path (None = auto-generate)
            fps: Frames per second
        """
        if not clips:
            return None
            
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_edit.xml"
        
        video_name = os.path.basename(video_path)
        
        # Create XML structure
        fcpxml = ET.Element("fcpxml", version="1.9")
        resources = ET.SubElement(fcpxml, "resources")
        library = ET.SubElement(fcpxml, "library")
        event = ET.SubElement(library, "event", name="AI Video Edit")
        project = ET.SubElement(event, "project", name="Edited Timeline")
        sequence = ET.SubElement(project, "sequence", format="r1")
        
        # Add format
        format_elem = ET.SubElement(resources, "format", 
                                   id="r1",
                                   name="FFVideoFormat1080p2997",
                                   frameDuration="1001/30000",
                                   width="1920",
                                   height="1080")
        
        # Add asset
        asset_id = f"asset-{hash(video_path) % 10000}"
        asset = ET.SubElement(resources, "asset",
                            id=asset_id,
                            name=video_name,
                            src=f"file://{video_path}")
        
        # Media duration
        duration_sec = clips[-1][1] - clips[0][0] if clips else 60
        duration_frames = int(duration_sec * fps)
        
        # Add sequence
        spine = ET.SubElement(sequence, "spine")
        
        # Total duration in frames
        total_duration = sum(end - start for start, end in clips)
        sequence.set("duration", f"{int(total_duration * fps * 100)}s")
        
        # Add each clip
        for i, (start, end) in enumerate(clips, 1):
            duration = end - start
            duration_frames = int(duration * fps * 100)
            start_frames = int(start * fps * 100)
            
            clip = ET.SubElement(spine, "clip",
                               name=f"Clip {i}",
                               duration=f"{duration_frames}s",
                               start=f"{start_frames}s")
            
            # Add video
            video = ET.SubElement(clip, "video")
            ET.SubElement(video, "offset", relative="start", value=f"{start_frames}s")
            
            # Add audio
            audio = ET.SubElement(clip, "audio")
            ET.SubElement(audio, "offset", relative="start", value=f"{start_frames}s")
            
            ET.SubElement(clip, "asset-ref", id=asset_id)
        
        # Pretty print
        xml_str = ET.tostring(fcpxml, encoding='utf-8')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove XML declaration if minidom adds it weird
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    @staticmethod
    def get_export_formats():
        """Return list of available export formats"""
        return [
            ("EDL (CMX3600)", "*.edl"),
            ("FCPXML (DaVinci Resolve)", "*.xml"),
            ("CSV", "*.csv"),
            ("JSON", "*.json")
        ]
    
    @staticmethod
    def export_auto(clips, video_path, format='edl'):
        """Auto-export based on format name"""
        format = format.lower()
        if format == 'edl':
            return TimelineExporter.to_edl(clips, video_path)
        elif format in ('fcpxml', 'xml', 'fcp'):
            return TimelineExporter.to_fcp_xml(clips, video_path)
        else:
            return None