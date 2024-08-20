#pragma warning disable OPENAI001
using System.ClientModel;
using System.Reflection.Emit;
using Azure;
using Azure.AI.OpenAI;
using OpenAI;
using OpenAI.Assistants;
using OpenAI.Files;
using OpenAI.VectorStores;

namespace AzureOpenAiFileSearch;

public class Worker(ILogger<Worker> logger, IConfiguration configuration) : BackgroundService
{
    private readonly ILogger<Worker> _logger = logger;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var client = GetClient();
        var assistantClient = client.GetAssistantClient();
        var vectorStoreClient = client.GetVectorStoreClient();
        var fileClient = client.GetFileClient();

        VectorStore vectorStore = await CreateVectorStore(vectorStoreClient, stoppingToken);
        OpenAIFileInfo file = await UploadFileAsync(client, vectorStore, fileClient);
        AssistantThread thread = await assistantClient.CreateThreadAsync(cancellationToken: stoppingToken);
        Assistant assistant = await CreateAssistantAsync(assistantClient, vectorStore,  stoppingToken);

        while (true)
        {
            var request = ReadConsole();

            if (request.ToLower() == "exit")
            {
                break;
            }

            await assistantClient.CreateMessageAsync(thread.Id, [MessageContent.FromText(request)], cancellationToken: stoppingToken);
        
            ThreadRun threadRun = await assistantClient.CreateRunAsync(thread.Id, assistant.Id, cancellationToken: stoppingToken);
        
            do
            {
                await Task.Delay(TimeSpan.FromMilliseconds(100));
                threadRun = await assistantClient.GetRunAsync(thread.Id, threadRun.Id);
            }
            while (threadRun.Status == RunStatus.Queued || threadRun.Status == RunStatus.InProgress);
        
            AsyncPageableCollection<ThreadMessage> messagePage = assistantClient.GetMessagesAsync(thread.Id, ListOrder.NewestFirst);
            await using var enumerator = messagePage.GetAsyncEnumerator(stoppingToken);
            var messageItem = await enumerator.MoveNextAsync() ? enumerator.Current : null;
            
            foreach (MessageContent contentItem in messageItem!.Content)
            {
                if (!string.IsNullOrEmpty(contentItem.Text))
                {
                    Console.WriteLine($"Assistants -> {contentItem.Text}");

                    if (contentItem.TextAnnotations.Count > 0)
                    {
                        Console.WriteLine();
                    }
                    
                    foreach (TextAnnotation annotation in contentItem.TextAnnotations)
                    {
                        if (!string.IsNullOrEmpty(annotation.InputFileId))
                        {
                            Console.WriteLine($"->  File citation, file ID: {annotation.InputFileId}");
                        }
                    }
                }
            } 
            
            Console.WriteLine($"-> Completions tokens: {threadRun.Usage.CompletionTokens}");
            Console.WriteLine($"-> PromptTokens tokens: {threadRun.Usage.PromptTokens}");
            Console.WriteLine($"-> TotalTokens tokens: {threadRun.Usage.TotalTokens}");
            Console.WriteLine();
        }

        await fileClient.DeleteFileAsync(file);
        await vectorStoreClient.DeleteVectorStoreAsync(vectorStore);
        await assistantClient.DeleteThreadAsync(thread);
        await assistantClient.DeleteAssistantAsync(assistant);
    }

    private async Task<OpenAIFileInfo> UploadFileAsync(AzureOpenAIClient client, VectorStore vectorStore, FileClient fileClient)
    {
        // Upload file
        var filePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "History.md");
        
        OpenAIFileInfo file =await fileClient.UploadFileAsync(filePath, FileUploadPurpose.Assistants);
        
        // Link file to the vector store
        var vectorStoreClient = client.GetVectorStoreClient();
        await vectorStoreClient.AddFileToVectorStoreAsync(vectorStore, file);

        return file;
    }

    private async Task<VectorStore> CreateVectorStore(VectorStoreClient vectorStoreClient, CancellationToken stoppingToken)
    {
        // Create vector store
        return await vectorStoreClient.CreateVectorStoreAsync(new VectorStoreCreationOptions
        {
            Name = "Docs"
        }, stoppingToken);
    }
    
    private async Task<Assistant> CreateAssistantAsync(AssistantClient assistantClient, VectorStore vectorStore, CancellationToken stoppingToken)
    {
        // Create assistant
        var assistantOptions = new AssistantCreationOptions
        {
            Name = "my-assistant",
            Instructions = "You are a helpful assistant that can help fetch data from files you know about",
            Tools = { new FileSearchToolDefinition() },
            ToolResources = new ToolResources()
            {
                
                FileSearch = new FileSearchToolResources
                {
                    VectorStoreIds = [vectorStore.Id]
                }
            }
        };
        
        return await assistantClient.CreateAssistantAsync(configuration["AzureOpenAi:Deployment"]!, assistantOptions, stoppingToken);
    }

    private string ReadConsole()
    {
        Console.Write($"User -> ");
        return Console.ReadLine()!;
    }
    
    private AzureOpenAIClient GetClient()
    {
        return new AzureOpenAIClient(
            new Uri(configuration["AzureOpenAi:Endpoint"]!), 
            new AzureKeyCredential(configuration["AzureOpenAi:Key"]!));
    }
}
#pragma warning restore OPENAI001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.