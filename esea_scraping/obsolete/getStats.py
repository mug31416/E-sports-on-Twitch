from itertools import islice
from twitch import TwitchHelix
from twitch import TwitchClient

#channel = client.channels.get_by_id(44322889)
#print(channel.id)
#print(channel.name)
#print(channel.display_name)


client = TwitchHelix(client_id='x2nlpgpmb4nn6m1o1vby49mif3qeqy')
streams_iterator = client.get_streams(page_size=100)
for stream in islice(streams_iterator, 0, 5):
    print(stream)

client = TwitchClient(client_id='x2nlpgpmb4nn6m1o1vby49mif3qeqy')
users = client.users.translate_usernames_to_ids(['vektv', 'jukes','officialf0xey'])
for user in users:
  print('{}: {}'.format(user.name, user.id))


client = TwitchHelix(client_id='x2nlpgpmb4nn6m1o1vby49mif3qeqy')
streams_iterator = client.get_streams_metadata(page_size=100,user_ids=[77208443])
for stream in islice(streams_iterator, 0, 5):
    print(stream)

client = TwitchHelix(client_id='x2nlpgpmb4nn6m1o1vby49mif3qeqy')
games_iterator = client.get_top_games(page_size=100)
for game in islice(games_iterator, 0, 10):
    print(game)

print("\n")
client = TwitchHelix(client_id='x2nlpgpmb4nn6m1o1vby49mif3qeqy')
#games_iterator = client.get_games(game_ids=['509658'])#names=['Counter-Strike: Global Offensive'])
games_iterator = client.get_games(names=["Counter-Strike: Global Offensive","FIFA 19","NHL 19"])
for game in islice(games_iterator, 0, 10):
    print(game)

from twitch import TwitchClient
client = TwitchClient('x2nlpgpmb4nn6m1o1vby49mif3qeqy')
teams = client.teams.get_all()
for t in islice(teams, 0, 10):
    print(t)