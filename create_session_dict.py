from datetime import datetime
import math

file = open("clicks_and_buys_partial.dat", "r")
lines = (line.split(',') for line in file.readlines())
file.close()
session_to_eventlist = {}

def onehotstring(i, N):
	### e.g. ohs(5,12) would return '000010000000'
	beg = '0'*(i-1)
	end = '0'*(N-i)
	return beg+'1'+end

print("Creating session info...")
counter = 0
prodIDset = set()
for line in lines:
	if counter%500000==0: print('\t'+str(counter))
	sessionID = line[0]
	# pad with zeros
	sessionID = '0'*(8-len(sessionID)) + sessionID
	dt = line[1]
	prodID = line[2]
	prodIDset.add(prodID)
	if sessionID not in session_to_eventlist:
		session_to_eventlist[sessionID] = []
	year = dt[0:4]
	month = dt[5:7]
	day = dt[8:10]
	hour = dt[11:13]
	minute = dt[14:16]
	second = dt[17:19]
	click_or_buy = '0' if counter <= 33003943 else '1'
	dt_obj = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
	wkday = str(dt_obj.weekday())
	stamp = dt_obj.timestamp()
	session_to_eventlist[sessionID].append((prodID, year, month, day, wkday, hour, minute, second, stamp, click_or_buy))
	counter += 1

print("Number of sessionIDs:", len(session_to_eventlist))

prodIDlist = sorted([x for x in prodIDset], key=lambda x: int(x))
print(len(prodIDlist))
max_prodID_num_size = len(str(len(prodIDlist)))
prodID_to_num = {pID: ('0'*(max_prodID_num_size-len(str(i+1))))+str(i+1) for i, pID in enumerate(prodIDlist)}



print("Sorting eventlists...")
for sessionID in session_to_eventlist:
	session_to_eventlist[sessionID] = sorted(session_to_eventlist[sessionID], key=lambda tup: tup[-2])


print("Adding time diffs...")
# add time diffs
counter = 0
for sessionID in session_to_eventlist:
	if counter%500000==0: print('\t'+str(counter))
	L = session_to_eventlist[sessionID]
	new_L = []
	for i in range(len(L)):
		event_tuple = L[i]
		abs_diff = 0.0 if i == 0 else event_tuple[-2] - L[i-1][-2]
		time_diff = "00" if (counter==0 or abs_diff < 1.0) else str(math.ceil(math.log(abs_diff, 2)))
		if int(time_diff) > 20:
			time_diff = '20'
		# pad time diff with zero if necessary
		if len(time_diff) == 1:
			time_diff = '0'+time_diff
		new_event_tuple = event_tuple + (time_diff,)
		new_L.append(new_event_tuple)
	session_to_eventlist[sessionID] = new_L
	counter += 1


print("Creating formatted event vector pairs and ensuring the context info lists are size 20...")
# https://en.wikipedia.org/wiki/Procrustes   -- relevant
session_to_contextinfo = {}
empty_contextinfo = ('0'*10, '0'*66)
counter = 0
for sessionID in session_to_eventlist:
	if counter%500000==0: print('\t'+str(counter))
	if sessionID not in session_to_contextinfo:
		session_to_contextinfo[sessionID] = []
	for event_tuple in session_to_eventlist[sessionID]:
		prodID, year, month, day, wkday, hour, minute, second, stamp, click_or_buy, time_diff = event_tuple
		# one-hot-style
		session_to_contextinfo[sessionID].append((prodID_to_num[prodID],  onehotstring(int(month),12) + onehotstring(int(hour)+1, 24)+ onehotstring(int(wkday)+1, 7) + onehotstring(int(time_diff)+1, 21) + onehotstring(int(click_or_buy)+1, 2)))
	m = len(session_to_contextinfo[sessionID])
	if m < 20:
		session_to_contextinfo[sessionID] += [empty_contextinfo]*(20-len(session_to_contextinfo[sessionID]))
	if m > 20:
		session_to_contextinfo[sessionID] = session_to_contextinfo[sessionID][-20:]
	counter += 1


print("Writing to final file now...")
final_out_file = open("final_out_file_partial.dat", "w")

sessionIDs_sorted = sorted([k for k in session_to_contextinfo.keys()], key=lambda k: int(k))
counter = 0
for sessionID in sessionIDs_sorted:
	if counter%500000==0: print('\t'+str(counter))
	if session_to_contextinfo[sessionID][1][0] != '0000000000': # if sessionID has more than one event
		final_out_file.write(sessionID + ' ')
		for event in session_to_contextinfo[sessionID][:-1]:
			final_out_file.write(event[0] + ' ' + event[1] + ' ')
		final_out_file.write(session_to_contextinfo[sessionID][-1][0] + ' ' + session_to_contextinfo[sessionID][-1][1]+'\n')
	counter += 1


final_out_file.close()



