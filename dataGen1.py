from __future__ import unicode_literals
import youtube_dl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import csv
import ast
import os.path
#Class definitions

em_class = { '/m/03j1ly':'Emergency vehicle',
		'/m/04qvtq':'Police car(siren)',
		'/m/012n7d':'Ambulance(siren)',
		'/m/012ndj':'Fire engine, fire truck(siren)'
		#'/m/03kmc9':'Siren'		
}
non_em_class = {'/m/07pbtc8':'Walk, footsteps',
		'/m/0gy1t2s':'Bicycle bell',
		'/m/03m9d0z':'Wind',
		'/t/dd00092':'Wind noise(microphone)',
		'/m/0jb2l'  :'Thunderstorm',
		'/m/0ngt1'  :'Thunder',
		'/m/06mb1'  :'Rain',
		'/m/0j2kx'  :'Waterfall',
		##can probably exclude this
		'/m/028v0c' :'Silence',
		'/m/096m7z' :'Noise',
		'/m/06_y0by':'Environmental noise',
		'/m/07rgkc5':'Static',
		'/m/0chx_'  :'White noise',
		'/m/06bz3'  :'Radio',		
		#could be noise classes those listed above
		
		###not sure about this '/m/07yv9'  :'Vehicle',
		###not sure about this '/m/012f08' :'Moto vehicle(road)',
		'/m/0k4j'   :'Car',
		'/m/0912c9' :'Vehicle horn, car horn, honking',
		'/m/07rknqz':'Skidding',
		'/m/0h9mv'  :'Tire squeal',
		'/t/dd00134':'Car passing by',
		'/m/07r04'  :'Truck',
		'/m/05x_td' :'Air horn,truck horn',
		'/m0/4_sv'  :'Motorcycle',
	 	'/m/0btp2'  :'Traffic noise, roadway noise',
		'/m/0195fx' :'Subway, metro, underground',
		'/m/0199g'  :'Bicycle',
		'/m/06_fw'  :'Skateboard',
		
		##very important to classify these as non-em
		'/m/0dgbq'  :'Civil defense siren',
		'/m/030rvx' :'Buzzer',
		'/m/01y3hg' :'Smoke detector, smoke alarm',
		'/m/0c3f7m' :'Fire alarm'				
}

with open('balanced_train_segments.csv') as csvfile:
	readCSV = csv.reader(csvfile,delimiter=str(u',').encode('utf-8'))
	count=0
	em_c=138
	non_em_c=1748
	for row in readCSV:
		if(count<3):
			count=count+1;
		else:
			print 'hi'
			v_id = row[0]
			v_start = ast.literal_eval(row[1])
			v_end = ast.literal_eval(row[2])
			labels = row[3]
			
			flg1=0
			flg2=0
			v_class = labels.split(",")
			##Excluding videos present in both EM and non-EM
			if(set(v_class)&set(em_class.keys())!=set([])):
				flg1=1
			if(set(v_class)&set(non_em_class.keys())!=set([])):
				flg2=1
			if flg1==1 and flg2==1:
				continue;
			elif flg1==1:
				print v_id,v_start,v_end,'em'
				em_c = em_c+1
				ydl_opts = {
    					'format': 'wav/mp3/bestaudio/best',
					'ignoreerrors':'True',
    					'postprocessors': [{
        					'key': 'FFmpegExtractAudio',
        					'preferredcodec': 'wav',
        					'preferredquality': '192',
    					}],
    					'outtmpl':'eme/'+str(em_c)+'_m.%(ext)s',    
				}

				with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    					ydl.download(['http://www.youtube.com/watch?v='+row[0]])
				
				input_file = '/home/geethik/Desktop/eme/'+str(em_c)+'_m.wav'
				if(os.path.exists(input_file)):
					#em_c = em_c +1
					output_file = 'eme/'+str(em_c)+'.wav'
					ffmpeg_extract_subclip(input_file,v_start,v_end,targetname=output_file)
					os.remove(input_file)
				else:
					em_c = em_c-1
			elif flg2==1:
				print v_id,v_start,v_end,'Non-em'
				non_em_c = non_em_c+1
				ydl_opts = {
    					'format': 'wav/mp3/bestaudio/best',
					'ignoreerrors':'True',
    					'postprocessors': [{
        					'key': 'FFmpegExtractAudio',
        					'preferredcodec': 'wav',
        					'preferredquality': '192',
    					}],
    					'outtmpl':'nonEme/'+str(non_em_c)+'_m.%(ext)s',    
				}

				with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    					ydl.download(['http://www.youtube.com/watch?v='+row[0]])
				
				input_file = '/home/geethik/Desktop/nonEme/'+str(non_em_c)+'_m.wav'
				if(os.path.exists(input_file)):					
					output_file = 'nonEme/'+str(non_em_c)+'.wav'
					ffmpeg_extract_subclip(input_file,v_start,v_end,targetname=output_file)
					os.remove(input_file)	
				else:
					non_em_c = non_em_c-1






