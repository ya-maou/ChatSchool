import weaviate

client = weaviate.connect_to_local()

# 獲取所有集合配置
collections = client.collections.list_all()

print("Weaviate 中的 Collections（Class）有：")
for name in collections:
    print(name)
    
client.close()
