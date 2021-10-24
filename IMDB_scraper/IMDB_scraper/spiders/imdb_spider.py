# to run 
# scrapy crawl imdb_spider -o movies.csv
import scrapy


class ImdbSpider(scrapy.Spider):
	name = 'imdb_spider'
	
	start_urls = ['https://www.imdb.com/title/tt2382320/']


	def parse(self, response):

		Tile = "fullcredits"

		CrewAndCast = response.url + Tile

		yield scrapy.Request(CrewAndCast, callback = self.parse_full_credits)


	def parse_full_credits(self, response):

		for actor_link in [a.attrib["href"] for a in response.css("td.primary_photo a")]:

			if actor_link:
				actor_link = response.urljoin(actor_link)

			yield scrapy.Request(actor_link, callback = self.parse_actor_page)


	def parse_actor_page(self, response):
		actor_name = response.css("span.itemprop::text").get()

		for MOVIES in response.css("div.filmo-row"):
			movie_or_TV_name = [MOVIES.css("a::text").get()]

			yield {
					"actor" : actor_name,
					"movie_or_TV_name" : movie_or_TV_name
					}