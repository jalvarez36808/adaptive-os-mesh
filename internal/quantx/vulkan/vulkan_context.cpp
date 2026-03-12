//go:build !darwin
#ifdef USE_VULKAN
#include "vulkan_context.hpp"
#include "quantx_vulkan_spv.h"
#include <iostream>
#include <cstring>

VulkanContext::VulkanContext() : 
    instance(VK_NULL_HANDLE), device(VK_NULL_HANDLE), 
    inputBuffer(VK_NULL_HANDLE), outputBuffer(VK_NULL_HANDLE),
    current_max_k(0) {}

VulkanContext::~VulkanContext() {
    destroyBuffers();
    if (device != VK_NULL_HANDLE) {
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorSetPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

void VulkanContext::destroyBuffers() {
    if (device == VK_NULL_HANDLE) return;
    if (inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        vkFreeMemory(device, inputBufferMemory, nullptr);
        inputBuffer = VK_NULL_HANDLE;
    }
    if (outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, outputBuffer, nullptr);
        vkFreeMemory(device, outputBufferMemory, nullptr);
        outputBuffer = VK_NULL_HANDLE;
    }
}

bool VulkanContext::init() {
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "QuantX", 1, "Vextra", 1, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &appInfo, 0, nullptr, 0, nullptr};
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return false;

    uint32_t count = 0; vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count); vk EnumeratePhysicalDevices(instance, &count, devices.data());
    for (auto d : devices) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { physicalDevice = d; break; }
    }
    if (!physicalDevice && count > 0) physicalDevice = devices[0];

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count); vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    for (uint32_t i = 0; i < count; i++) { if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeQueueFamilyIndex = i; break; } }

    float qp = 1.0f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, computeQueueFamilyIndex, 1, &qp};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &qi, 0, nullptr, 0, nullptr, nullptr};
    if (vkCreateDevice(physicalDevice, &di, nullptr, &device) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, b};
    vkCreateDescriptorSetLayout(device, &lci, nullptr, &descriptorSetLayout);

    VkShaderModuleCreateInfo sci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, (size_t)quantx_vulkan_spv_len, (const uint32_t*)quantx_vulkan_spv};
    vkCreateShaderModule(device, &sci, nullptr, &shaderModule);
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &descriptorSetLayout, 0, nullptr};
    vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout);
    VkComputePipelineCreateInfo pci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", nullptr}, pipelineLayout, VK_NULL_HANDLE, 0};
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo ppi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &ps};
    vkCreateDescriptorPool(device, &ppi, nullptr, &descriptorSetPool);
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorSetPool, 1, &descriptorSetLayout};
    vkAllocateDescriptorSets(device, &ai, &descriptorSet);

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeQueueFamilyIndex};
    vkCreateCommandPool(device, &cpci, nullptr, &commandPool);
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    return true;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps; vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) return i;
    }
    return -1;
}

bool VulkanContext::prepare(int max_k) {
    if (max_k <= current_max_k) return true;
    destroyBuffers();
    size_t inSize = (max_k / 256) * 72;
    size_t outSize = max_k * 4;

    auto alloc = [&](size_t s, VkBuffer& b, VkDeviceMemory& m, VkMemoryPropertyFlags p) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, s, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        vkCreateBuffer(device, &bi, nullptr, &b);
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, b, &mr);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, findMemoryType(mr.memoryTypeBits, p)};
        vkAllocateMemory(device, &ai, nullptr, &m);
        vkBindBufferMemory(device, b, m, 0);
    };

    // Use DEVICE_LOCAL for peak performance
    alloc(inSize, inputBuffer, inputBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    alloc(outSize, outputBuffer, outputBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDescriptorBufferInfo dbi[2] = {{inputBuffer, 0, inSize}, {outputBuffer, 0, outSize}};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi[0], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi[1], nullptr}
    };
    vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    current_max_k = max_k;
    return true;
}

bool VulkanContext::run_kernel(int k) {
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb; vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(cb, k / 256, 1, 1);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
    vkQueueSubmit(computeQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cb);
    return true;
}

bool VulkanContext::dequantize(const void* vx, float* vy, int k) {
    if (k > current_max_k && !prepare(k)) return false;
    size_t inSize = (k / 256) * 72;
    size_t outSize = k * 4;

    // Staging buffer for upload
    VkBuffer stagingIn; VkDeviceMemory stagingInMem;
    VkBufferCreateInfo sbi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, inSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    vkCreateBuffer(device, &sbi, nullptr, &stagingIn);
    VkMemoryRequirements smr; vkGetBufferMemoryRequirements(device, stagingIn, &smr);
    VkMemoryAllocateInfo sai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, smr.size, findMemoryType(smr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    vkAllocateMemory(device, &sai, nullptr, &stagingInMem);
    vkBindBufferMemory(device, stagingIn, stagingInMem, 0);
    void* p; vkMapMemory(device, stagingInMem, 0, inSize, 0, &p); std::memcpy(p, vx, inSize); vkUnmapMemory(device, stagingInMem);

    // Staging buffer for download
    VkBuffer stagingOut; VkDeviceMemory stagingOutMem;
    sbi.size = outSize; sbi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(device, &sbi, nullptr, &stagingOut);
    vkGetBufferMemoryRequirements(device, stagingOut, &smr);
    sai.allocationSize = smr.size; sai.memoryTypeIndex = findMemoryType(smr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &sai, nullptr, &stagingOutMem);
    vkBindBufferMemory(device, stagingOut, stagingOutMem, 0);

    // Copy, Run, Copy back
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb; vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy bc = {0, 0, inSize}; vkCmdCopyBuffer(cb, stagingIn, inputBuffer, 1, &bc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(cb, k / 256, 1, 1);
    bc.size = outSize; vkCmdCopyBuffer(cb, outputBuffer, stagingOut, 1, &bc);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
    vkQueueSubmit(computeQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    vkMapMemory(device, stagingOutMem, 0, outSize, 0, &p); std::memcpy(vy, p, outSize); vkUnmapMemory(device, stagingOutMem);

    vkFreeCommandBuffers(device, commandPool, 1, &cb);
    vkDestroyBuffer(device, stagingIn, nullptr); vkFreeMemory(device, stagingInMem, nullptr);
    vkDestroyBuffer(device, stagingOut, nullptr); vkFreeMemory(device, stagingOutMem, nullptr);
    return true;
}

extern "C" {
    void* vulkan_init() {
        VulkanContext* ctx = new VulkanContext();
        if (ctx->init()) return ctx;
        delete ctx; return nullptr;
    }
    int vulkan_prepare(void* ctx, int max_k) { return static_cast<VulkanContext*>(ctx)->prepare(max_k) ? 0 : 1; }
    int vulkan_run_kernel(void* ctx, int k) { return static_cast<VulkanContext*>(ctx)->run_kernel(k) ? 0 : 1; }
    int vulkan_dequantize(void* ctx, const void* vx, float* vy, int k) { return static_cast<VulkanContext*>(ctx)->dequantize(vx, vy, k) ? 0 : 1; }
    void vulkan_free(void* ctx) { delete static_cast<VulkanContext*>(ctx); }
}
#endif
