"""
동영상 파일 복사 도구
Downloads/archive에서 Bi-Direction 프로젝트로 동영상 복사
"""
import os
import shutil
import glob
from pathlib import Path


def copy_videos_from_archive():
    """Downloads/archive에서 동영상 파일들을 Bi-Direction/data/videos로 복사"""
    
    # 소스 디렉토리 (권한 문제로 직접 접근이 어려울 수 있음)
    source_dir = "/Users/hayubin/Downloads/archive"
    
    # 대상 디렉토리
    target_dir = "/Users/hayubin/Bi-Direction/data/videos"
    
    # 대상 디렉토리 생성
    os.makedirs(target_dir, exist_ok=True)
    
    # 지원하는 비디오 포맷
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    copied_count = 0
    
    print("🎥 Copying videos from Downloads/archive...")
    print("=" * 50)
    
    try:
        # 각 확장자별로 파일 검색
        for ext in video_extensions:
            pattern = os.path.join(source_dir, ext)
            files = glob.glob(pattern)
            
            # 대문자 확장자도 검색
            pattern_upper = os.path.join(source_dir, ext.upper())
            files.extend(glob.glob(pattern_upper))
            
            for video_file in files:
                filename = os.path.basename(video_file)
                target_path = os.path.join(target_dir, filename)
                
                try:
                    shutil.copy2(video_file, target_path)
                    print(f"✅ Copied: {filename}")
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Failed to copy {filename}: {e}")
    
    except Exception as e:
        print(f"❌ Error accessing source directory: {e}")
        print("💡 Try running this script manually or check permissions")
    
    print(f"\n📊 Total videos copied: {copied_count}")
    
    if copied_count > 0:
        print(f"📁 Videos saved to: {target_dir}")
        print("\n🚀 Next steps:")
        print("1. Run: python tools/video_processor.py")
        print("2. This will extract frames and create YOLO dataset")
    else:
        print("\n💡 No videos found. Please check:")
        print("1. Videos are in Downloads/archive/")
        print("2. File permissions are correct")
        print("3. Video files have supported extensions")


if __name__ == "__main__":
    copy_videos_from_archive()

